#!/usr/bin/env python3
"""Autoregressive decoder for product factor generation.

The decoder is a standard encoder-decoder Transformer decoder:
- target tokens use causal self-attention;
- every layer cross-attends Q-Former memory `[B, M, 768]` as key/value;
- loss is computed only on answer tokens, with padding labels set to -100.

Tokenizer convention uses the copied Chinese RoBERTa tokenizer:
- [PAD] -> pad
- [CLS] -> bos
- [SEP] -> eos
- [UNK] -> unk
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_DECODER_TOKENIZER_PATH = Path("/mnt/disk5/syh_flowwif2/Attribution/model_factors/decoder_tokenizer")
DEFAULT_ROBERTA_PATH = Path("/mnt/disk5/syh_flowwif2/Attribution/model_factors/chinese-roberta-wwm-ext")


@dataclass
class FactorDecoderBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_text: list[str]


@dataclass
class FactorDecoderOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    hidden_states: torch.Tensor
    cross_attentions: tuple[torch.Tensor, ...]


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic positional encoding to avoid tokenizer/model coupling."""

    def __init__(self, hidden_dim: int, max_position_embeddings: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_position_embeddings, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_position_embeddings, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        end = int(position_offset) + seq_len
        if end > self.pe.size(1):
            raise ValueError(f"Position end {end} exceeds max positional length {self.pe.size(1)}")
        return x + self.pe[:, position_offset:end].to(dtype=x.dtype, device=x.device)


class FactorDecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
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
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor,
        causal_mask: torch.Tensor | None,
        target_key_padding_mask: torch.Tensor | None = None,
        encoder_key_padding_mask: torch.Tensor | None = None,
        need_cross_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.self_attn_norm(hidden_states)
        self_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self_out

        x = self.cross_attn_norm(hidden_states)
        cross_out, cross_weights = self.cross_attn(
            query=x,
            key=encoder_memory,
            value=encoder_memory,
            key_padding_mask=encoder_key_padding_mask,
            need_weights=need_cross_weights,
            average_attn_weights=False,
        )
        hidden_states = hidden_states + cross_out
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, cross_weights

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.self_attn_norm(hidden_states)
        if past_key_value is None:
            key = value = x
        else:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, x], dim=1)
            value = torch.cat([past_value, x], dim=1)
        self_out, _ = self.self_attn(
            query=x,
            key=key,
            value=value,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )
        hidden_states = hidden_states + self_out

        y = self.cross_attn_norm(hidden_states)
        cross_out, _ = self.cross_attn(
            query=y,
            key=encoder_memory,
            value=encoder_memory,
            key_padding_mask=encoder_key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + cross_out
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, (key, value)


class FactorAutoregressiveDecoder(nn.Module):
    """12-layer AR decoder cross-attending Q-Former memory."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        pad_token_id: int = 0,
        tie_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.pad_token_id = int(pad_token_id)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, max_position_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            FactorDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_embedding:
            self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].zero_()

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_cross_attentions: bool = False,
    ) -> FactorDecoderOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,L], got {tuple(input_ids.shape)}")
        if encoder_memory.ndim != 3:
            raise ValueError(f"encoder_memory must be [B,M,D], got {tuple(encoder_memory.shape)}")
        if encoder_memory.shape[0] != input_ids.shape[0]:
            raise ValueError("encoder_memory batch size must match input_ids")
        if encoder_memory.shape[-1] != self.hidden_dim:
            raise ValueError(f"encoder_memory hidden dim must be {self.hidden_dim}, got {encoder_memory.shape[-1]}")

        target_key_padding_mask = None
        if attention_mask is not None:
            target_key_padding_mask = ~attention_mask.bool()
        encoder_key_padding_mask = None
        if encoder_attention_mask is not None:
            encoder_key_padding_mask = ~encoder_attention_mask.bool()

        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        causal_mask = self.build_causal_mask(input_ids.size(1), input_ids.device)

        cross_attentions: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, cross_weights = layer(
                hidden_states=hidden_states,
                encoder_memory=encoder_memory,
                causal_mask=causal_mask,
                target_key_padding_mask=target_key_padding_mask,
                encoder_key_padding_mask=encoder_key_padding_mask,
                need_cross_weights=return_cross_attentions,
            )
            if return_cross_attentions and cross_weights is not None:
                cross_attentions.append(cross_weights)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return FactorDecoderOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            cross_attentions=tuple(cross_attentions),
        )

    @torch.no_grad()
    def init_token_embedding_from_roberta(self, roberta_path: str | Path) -> None:
        roberta = AutoModel.from_pretrained(str(Path(roberta_path).expanduser().resolve()), torch_dtype=torch.float32)
        roberta_embedding = roberta.embeddings.word_embeddings.weight.detach()
        if roberta_embedding.shape != self.token_embedding.weight.shape:
            raise ValueError(
                f"RoBERTa embedding shape {tuple(roberta_embedding.shape)} does not match "
                f"decoder embedding {tuple(self.token_embedding.weight.shape)}"
            )
        with torch.no_grad():
            self.token_embedding.weight.copy_(roberta_embedding.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype))
            if self.lm_head.weight is not self.token_embedding.weight:
                self.lm_head.weight.copy_(roberta_embedding.to(device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype))
            if self.pad_token_id is not None:
                self.token_embedding.weight[self.pad_token_id].zero_()
        del roberta

    @torch.no_grad()
    def generate(
        self,
        encoder_memory: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 128,
        encoder_attention_mask: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        batch_size = encoder_memory.size(0)
        input_ids = torch.full(
            (batch_size, 1),
            fill_value=bos_token_id,
            dtype=torch.long,
            device=encoder_memory.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=encoder_memory.device)
        encoder_key_padding_mask = None
        if encoder_attention_mask is not None:
            encoder_key_padding_mask = ~encoder_attention_mask.bool()
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.layers)
        cur_token = input_ids
        for step in range(max_new_tokens):
            hidden_states = self.token_embedding(cur_token)
            hidden_states = self.position_encoding(hidden_states, position_offset=step)
            hidden_states = self.dropout(hidden_states)
            new_past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx, layer in enumerate(self.layers):
                hidden_states, kv = layer.forward_step(
                    hidden_states=hidden_states,
                    encoder_memory=encoder_memory,
                    past_key_value=past_key_values[layer_idx],
                    encoder_key_padding_mask=encoder_key_padding_mask,
                )
                new_past_key_values.append(kv)
            past_key_values = new_past_key_values
            hidden_states = self.final_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_token_id)
            cur_token = next_token[:, None]
            if finished.all():
                break
        return input_ids


class FactorDecoderTokenizer:
    """Tokenizer helper using copied Chinese RoBERTa WordPiece vocabulary."""

    def __init__(self, tokenizer_path: str | Path = DEFAULT_DECODER_TOKENIZER_PATH, max_length: int = 128) -> None:
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        self.max_length = int(max_length)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.bos_token_id = int(self.tokenizer.cls_token_id)
        self.eos_token_id = int(self.tokenizer.sep_token_id)
        self.unk_token_id = int(self.tokenizer.unk_token_id)
        if self.tokenizer.pad_token != "[PAD]" or self.tokenizer.cls_token != "[CLS]" or self.tokenizer.sep_token != "[SEP]":
            raise ValueError("Unexpected decoder tokenizer special token mapping")

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def encode_target_texts(self, target_texts: list[str], device: torch.device | str | None = None) -> FactorDecoderBatch:
        encoded = self.tokenizer(
            target_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        if ids.size(1) < 2:
            raise ValueError("Encoded target sequence must contain at least BOS and EOS tokens")

        input_ids = ids[:, :-1].contiguous()
        attention_mask = mask[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous()
        label_mask = mask[:, 1:].bool()
        labels = labels.masked_fill(~label_mask, -100)
        if device is not None:
            device = torch.device(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
        return FactorDecoderBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            target_text=list(target_texts),
        )

    def encode_batch_from_dataset(self, batch: dict[str, Any], device: torch.device | str | None = None) -> FactorDecoderBatch:
        return self.encode_target_texts(list(batch["target_text"]), device=device)

    def decode(self, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()

    def decode_batch(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_ids]

    @staticmethod
    def factors_to_text(factors: list[str]) -> str:
        return json.dumps(factors, ensure_ascii=False, separators=(",", ":"))


def build_factor_decoder(
    tokenizer_path: str | Path = DEFAULT_DECODER_TOKENIZER_PATH,
    roberta_path_for_embedding_init: str | Path | None = None,
    hidden_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    ffn_dim: int = 3072,
    dropout: float = 0.1,
    max_position_embeddings: int = 512,
    tie_embedding: bool = True,
    init_from_roberta: bool = True,
) -> tuple[FactorAutoregressiveDecoder, FactorDecoderTokenizer]:
    tok = FactorDecoderTokenizer(tokenizer_path=tokenizer_path, max_length=max_position_embeddings)
    model = FactorAutoregressiveDecoder(
        vocab_size=tok.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tok.pad_token_id,
        tie_embedding=tie_embedding,
    )
    if init_from_roberta:
        roberta_path = Path(roberta_path_for_embedding_init or DEFAULT_ROBERTA_PATH).expanduser().resolve()
        roberta = AutoModel.from_pretrained(str(roberta_path), torch_dtype=torch.float32)
        roberta_embedding = roberta.embeddings.word_embeddings.weight.detach()
        if roberta_embedding.shape != model.token_embedding.weight.shape:
            raise ValueError(
                f"RoBERTa embedding shape {tuple(roberta_embedding.shape)} does not match "
                f"decoder embedding {tuple(model.token_embedding.weight.shape)}"
            )
        with torch.no_grad():
            model.token_embedding.weight.copy_(roberta_embedding)
            if model.lm_head.weight is not model.token_embedding.weight:
                model.lm_head.weight.copy_(roberta_embedding)
        del roberta
    return model, tok
