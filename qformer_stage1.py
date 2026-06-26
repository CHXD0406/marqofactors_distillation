#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class QFormerStage1Output:


    text_queries: torch.Tensor
    updated_text_tokens: torch.Tensor
    text_cross_attentions: tuple[torch.Tensor, ...]
    route_weights: tuple[torch.Tensor, ...]


class QFormerStage1Block(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_routes: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_routes = int(num_routes)

        self.route_queries = nn.Parameter(torch.empty(num_routes, num_queries, hidden_dim))
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)

        self.text_self_norm = nn.LayerNorm(hidden_dim)
        self.text_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_ffn_norm = nn.LayerNorm(hidden_dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.route_query_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_routes))
        self.route_cross_attns = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_routes)
        )
        self.route_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_routes),
        )
        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        prev_queries: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size = text_tokens.shape[0]
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.bool()

        t = self.text_self_norm(text_tokens)
        text_self_out, _ = self.text_self_attn(
            query=t,
            key=t,
            value=t,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        text_tokens = text_tokens + text_self_out
        text_tokens = text_tokens + self.text_ffn(self.text_ffn_norm(text_tokens))

        if text_attention_mask is None:
            text_summary = text_tokens.mean(dim=1)
        else:
            mask = text_attention_mask.to(dtype=text_tokens.dtype).unsqueeze(-1)
            text_summary = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        route_weights = torch.softmax(self.route_gate(text_summary), dim=-1)  # [B, R]

        route_outputs: list[torch.Tensor] = []
        route_attns: list[torch.Tensor] = []
        for route_idx, cross_attn in enumerate(self.route_cross_attns):
            query = self.route_queries[route_idx].unsqueeze(0).expand(batch_size, -1, -1)
            if prev_queries is not None:
                query = query + prev_queries
            query = self.route_query_norms[route_idx](query)
            route_out, route_attn = cross_attn(
                query=query,
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            route_outputs.append(route_out)
            if need_weights and route_attn is not None:
                route_attns.append(route_attn)

        stacked_outputs = torch.stack(route_outputs, dim=1)  # [B,R,Q,D]
        queries = (stacked_outputs * route_weights[:, :, None, None]).sum(dim=1)
        queries = queries + self.query_ffn(self.query_ffn_norm(queries))

        if need_weights and route_attns:
            stacked_attns = torch.stack(route_attns, dim=1)  # [B,R,H,Q,T]
            weighted_attn = (stacked_attns * route_weights[:, :, None, None, None]).sum(dim=1)
        else:
            weighted_attn = None

        return queries, text_tokens, weighted_attn, route_weights


class QFormerStage1TextDenoiser(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_routes: int = 4,
        num_heads: int = 12,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_routes = int(num_routes)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.layers = nn.ModuleList(
            QFormerStage1Block(
                hidden_dim=hidden_dim,
                num_queries=num_queries,
                num_routes=num_routes,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        return_attentions: bool = True,
    ) -> QFormerStage1Output:
        if text_tokens.ndim != 3:
            raise ValueError(f"text_tokens must be [B,T,D], got {tuple(text_tokens.shape)}")
        _, _, hidden_dim = text_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"QFormerStage1 expected text hidden dim {self.hidden_dim}, got {hidden_dim}. "
                "Chinese RoBERTa wwm-ext should output hidden_size=768."
            )
        if text_attention_mask is not None and text_attention_mask.shape[:2] != text_tokens.shape[:2]:
            raise ValueError(
                f"text_attention_mask shape {tuple(text_attention_mask.shape)} does not match "
                f"text_tokens [B,T] = {tuple(text_tokens.shape[:2])}"
            )

        updated_text_tokens = text_tokens
        attentions: list[torch.Tensor] = []
        route_weights_list: list[torch.Tensor] = []
        queries: torch.Tensor | None = None
        for layer in self.layers:
            queries, updated_text_tokens, attn, route_weights = layer(
                text_tokens=updated_text_tokens,
                prev_queries=queries,
                text_attention_mask=text_attention_mask,
                need_weights=return_attentions,
            )
            route_weights_list.append(route_weights)
            if return_attentions and attn is not None:
                attentions.append(attn)

        if queries is None:
            raise RuntimeError("QFormerStage1 has no layers; num_layers must be positive")
        queries = self.output_norm(queries)
        return QFormerStage1Output(
            text_queries=queries,
            updated_text_tokens=updated_text_tokens,
            text_cross_attentions=tuple(attentions),
            route_weights=tuple(route_weights_list),
        )


def build_qformer_stage1_text_denoiser(**kwargs: Any) -> QFormerStage1TextDenoiser:
    return QFormerStage1TextDenoiser(**kwargs)
