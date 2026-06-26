#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class QFormerStage2Output:
  

    product_queries: torch.Tensor
    all_queries: torch.Tensor
    image_tokens: torch.Tensor
    distill_attentions: dict[int, torch.Tensor]
    selected_layers: tuple[int, ...]
    selected_heads: tuple[int, ...]
    image_grid_h: int
    image_grid_w: int


class QFormerStage2Block(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.query_self_norm = nn.LayerNorm(hidden_dim)
        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_cross_norm = nn.LayerNorm(hidden_dim)
        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.query_self_norm(queries)
        self_out, _ = self.query_self_attn(
            query=q,
            key=q,
            value=q,
            need_weights=False,
        )
        queries = queries + self_out

        q = self.image_cross_norm(queries)
        cross_out, cross_attn = self.image_cross_attn(
            query=q,
            key=image_tokens,
            value=image_tokens,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        queries = queries + cross_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, cross_attn


class QFormerStage2ImageGrounder(nn.Module):

    def __init__(
        self,
        text_query_count: int = 32,
        extra_query_count: int = 8,
        image_dim: int = 1024,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        distill_layers: Iterable[int] = (2, 5, 7),
        distill_heads: Iterable[int] = tuple(range(10)),
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.text_query_count = int(text_query_count)
        self.extra_query_count = int(extra_query_count)
        self.total_query_count = self.text_query_count + self.extra_query_count
        self.image_dim = int(image_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.distill_layers = tuple(int(x) for x in distill_layers)
        self.distill_heads = tuple(int(x) for x in distill_heads)

        for layer_idx in self.distill_layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(f"distill layer {layer_idx} out of range for num_layers={self.num_layers}")
        for head_idx in self.distill_heads:
            if head_idx < 0 or head_idx >= self.num_heads:
                raise ValueError(f"distill head {head_idx} out of range for num_heads={self.num_heads}")

        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
        )
        self.extra_queries = nn.Parameter(torch.empty(extra_query_count, hidden_dim))
        nn.init.normal_(self.extra_queries, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            QFormerStage2Block(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_queries: torch.Tensor,
        image_tokens: torch.Tensor,
        image_grid_h: int = 37,
        image_grid_w: int = 37,
        return_distill_attentions: bool = True,
    ) -> QFormerStage2Output:
        if text_queries.ndim != 3:
            raise ValueError(f"text_queries must be [B,Q,D], got {tuple(text_queries.shape)}")
        if image_tokens.ndim != 3:
            raise ValueError(f"image_tokens must be [B,N,Dv], got {tuple(image_tokens.shape)}")
        batch_size, query_count, hidden_dim = text_queries.shape
        if query_count != self.text_query_count:
            raise ValueError(f"expected {self.text_query_count} text queries, got {query_count}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected text query dim {self.hidden_dim}, got {hidden_dim}")
        if image_tokens.shape[-1] != self.image_dim:
            raise ValueError(f"expected image token dim {self.image_dim}, got {image_tokens.shape[-1]}")
        if image_tokens.shape[1] != int(image_grid_h) * int(image_grid_w):
            raise ValueError(
                f"image token count {image_tokens.shape[1]} does not match grid "
                f"{image_grid_h}x{image_grid_w}"
            )

        projected_image_tokens = self.image_proj(image_tokens)
        extra_queries = self.extra_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = torch.cat([text_queries, extra_queries], dim=1)

        distill_attentions: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            need_weights = return_distill_attentions and layer_idx in self.distill_layers
            queries, image_attn = layer(
                queries=queries,
                image_tokens=projected_image_tokens,
                need_weights=need_weights,
            )
            if need_weights and image_attn is not None:
                distill_attentions[layer_idx] = image_attn[:, self.distill_heads, :, :].contiguous()

        queries = self.output_norm(queries)
        product_queries = queries[:, : self.text_query_count, :].contiguous()
        return QFormerStage2Output(
            product_queries=product_queries,
            all_queries=queries,
            image_tokens=projected_image_tokens,
            distill_attentions=distill_attentions,
            selected_layers=self.distill_layers,
            selected_heads=self.distill_heads,
            image_grid_h=int(image_grid_h),
            image_grid_w=int(image_grid_w),
        )


def build_qformer_stage2_image_grounder(**kwargs: Any) -> QFormerStage2ImageGrounder:
    """Factory with explicit name for training scripts."""
    return QFormerStage2ImageGrounder(**kwargs)
