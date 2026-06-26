#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .decoder import (
        DEFAULT_DECODER_TOKENIZER_PATH,
        DEFAULT_ROBERTA_PATH,
        FactorAutoregressiveDecoder,
        FactorDecoderOutput,
        FactorDecoderTokenizer,
    )
    from .encoders import ProductFactorFrozenEncoders, resolve_device
    from .qformer_stage1 import QFormerStage1TextDenoiser
    from .qformer_stage2 import QFormerStage2ImageGrounder
except ImportError:
    from decoder import (  # type: ignore
        DEFAULT_DECODER_TOKENIZER_PATH,
        DEFAULT_ROBERTA_PATH,
        FactorAutoregressiveDecoder,
        FactorDecoderOutput,
        FactorDecoderTokenizer,
    )
    from encoders import ProductFactorFrozenEncoders, resolve_device  # type: ignore
    from qformer_stage1 import QFormerStage1TextDenoiser  # type: ignore
    from qformer_stage2 import QFormerStage2ImageGrounder  # type: ignore

DEFAULT_TEXT_MODEL_PATH = Path("/mnt/disk5/syh_flowwif2/Attribution/model_factors/chinese-roberta-wwm-ext")
DEFAULT_VISION_MODEL_PATH = Path("/mnt/disk5/syh_flowwif2/Attribution/model_factors/dinov2-large")


@dataclass
class ProductFactorModelOutput:
    

    logits: torch.Tensor
    loss: torch.Tensor | None
    qformer_memory: torch.Tensor
    stage2_distill_attentions: dict[int, torch.Tensor]
    ce_loss: torch.Tensor | None = None
    generated_ids: torch.Tensor | None = None
    generated_text: list[str] | None = None


class ProductFactorModel(nn.Module):
  

    def __init__(
        self,
        text_model_path: str | Path = DEFAULT_TEXT_MODEL_PATH,
        vision_model_path: str | Path = DEFAULT_VISION_MODEL_PATH,
        tokenizer_path: str | Path = DEFAULT_DECODER_TOKENIZER_PATH,
        roberta_path_for_embedding_init: str | Path = DEFAULT_ROBERTA_PATH,
        text_max_length: int = 256,
        decoder_max_length: int = 128,
        device: torch.device | str | int | None = None,
        encoder_dtype: torch.dtype | None = None,
        init_decoder_from_roberta: bool = True,
        perm_ce_max_samples: int = 3,
        perm_ce_tau: float = 0.2,
        use_permutation_ce: bool = True,
    ) -> None:
        super().__init__()
        self.device_for_modules = resolve_device(device)
        self.encoders = ProductFactorFrozenEncoders(
            text_model_path=text_model_path,
            vision_model_path=vision_model_path,
            text_max_length=text_max_length,
            device=self.device_for_modules,
            dtype=encoder_dtype,
        )
        self.stage1 = QFormerStage1TextDenoiser()
        self.stage2 = QFormerStage2ImageGrounder()
        self.decoder_tokenizer = FactorDecoderTokenizer(tokenizer_path, max_length=decoder_max_length)
        self.perm_ce_max_samples = int(perm_ce_max_samples)
        self.perm_ce_tau = float(perm_ce_tau)
        self.use_permutation_ce = bool(use_permutation_ce)
        self.decoder = FactorAutoregressiveDecoder(
            vocab_size=self.decoder_tokenizer.vocab_size,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            max_position_embeddings=decoder_max_length,
            hidden_dim=768,
            num_layers=14,
            num_heads=12,
            ffn_dim=3072,
            dropout=0.1,
            tie_embedding=True,
        )
        self.to(self.device_for_modules)

        if init_decoder_from_roberta:
            self.decoder.init_token_embedding_from_roberta(roberta_path_for_embedding_init)

    @property
    def device(self) -> torch.device:
        return next(self.stage1.parameters()).device

    def encode_qformer(
        self,
        batch: dict[str, Any],
        return_stage1_attentions: bool = False,
        return_stage2_attentions: bool = True,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    
        with torch.no_grad():
            encoded = self.encoders(batch)

        text_tokens = encoded["text_tokens"].to(device=self.device, dtype=torch.float32)
        text_attention_mask = encoded["text_attention_mask"].to(device=self.device)
        image_tokens = encoded["image_tokens"].to(device=self.device, dtype=torch.float32)

        stage1_out = self.stage1(
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            return_attentions=return_stage1_attentions,
        )
        stage2_out = self.stage2(
            text_queries=stage1_out.text_queries,
            image_tokens=image_tokens,
            image_grid_h=int(encoded["image_grid_h"]),
            image_grid_w=int(encoded["image_grid_w"]),
            return_distill_attentions=return_stage2_attentions,
        )
        return stage2_out.all_queries, stage2_out.distill_attentions

    def _sample_factor_permutation_texts(self, factors: list[Any]) -> list[str]:
        clean_factors = [str(x) for x in factors if str(x).strip()]
        n = len(clean_factors)
        if n <= 3:
            sample_count = 1
        elif n <= 6:
            sample_count = min(2, self.perm_ce_max_samples)
        else:
            sample_count = min(3, self.perm_ce_max_samples)
        if sample_count <= 1 or n <= 1:
            return [json.dumps(clean_factors, ensure_ascii=False, separators=(",", ":"))]

        seen: set[tuple[str, ...]] = set()
        texts: list[str] = []
        original = tuple(clean_factors)
        seen.add(original)
        texts.append(json.dumps(list(original), ensure_ascii=False, separators=(",", ":")))
        max_attempts = sample_count * 8
        attempts = 0
        while len(texts) < sample_count and attempts < max_attempts:
            attempts += 1
            perm = clean_factors[:]
            random.shuffle(perm)
            key = tuple(perm)
            if key in seen:
                continue
            seen.add(key)
            texts.append(json.dumps(perm, ensure_ascii=False, separators=(",", ":")))
        return texts

    def _permutation_ce_loss(
        self,
        qformer_memory: torch.Tensor,
        batch: dict[str, Any],
        return_decoder_cross_attentions: bool = False,
    ) -> tuple[FactorDecoderOutput, torch.Tensor]:
        if "target_factors" not in batch:
            decoder_batch = self.decoder_tokenizer.encode_batch_from_dataset(batch, device=self.device)
            decoder_out = self.decoder(
                input_ids=decoder_batch.input_ids,
                encoder_memory=qformer_memory,
                attention_mask=decoder_batch.attention_mask,
                labels=decoder_batch.labels,
                encoder_attention_mask=None,
                return_cross_attentions=return_decoder_cross_attentions,
            )
            return decoder_out, decoder_out.loss if decoder_out.loss is not None else qformer_memory.new_tensor(0.0)

        all_texts: list[str] = []
        owner_indices: list[int] = []
        for sample_idx, factors in enumerate(batch["target_factors"]):
            texts = self._sample_factor_permutation_texts(list(factors))
            all_texts.extend(texts)
            owner_indices.extend([sample_idx] * len(texts))

        decoder_batch = self.decoder_tokenizer.encode_target_texts(all_texts, device=self.device)
        owner = torch.tensor(owner_indices, dtype=torch.long, device=self.device)
        expanded_memory = qformer_memory.index_select(0, owner)
        decoder_out = self.decoder(
            input_ids=decoder_batch.input_ids,
            encoder_memory=expanded_memory,
            attention_mask=decoder_batch.attention_mask,
            labels=None,
            encoder_attention_mask=None,
            return_cross_attentions=return_decoder_cross_attentions,
        )
        token_losses = F.cross_entropy(
            decoder_out.logits.reshape(-1, self.decoder.vocab_size),
            decoder_batch.labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(decoder_batch.labels.shape)
        valid = decoder_batch.labels.ne(-100)
        seq_losses = (token_losses * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)

        per_sample_losses: list[torch.Tensor] = []
        for sample_idx in range(qformer_memory.size(0)):
            losses_i = seq_losses[owner.eq(sample_idx)]
            if losses_i.numel() == 1:
                per_sample_losses.append(losses_i[0])
            else:
                tau = max(self.perm_ce_tau, 1e-6)
                soft_min = -tau * (torch.logsumexp(-losses_i / tau, dim=0) - torch.log(losses_i.new_tensor(losses_i.numel(), dtype=losses_i.dtype)))
                per_sample_losses.append(soft_min)
        loss = torch.stack(per_sample_losses).mean()
        decoder_out = FactorDecoderOutput(
            logits=decoder_out.logits,
            loss=loss,
            hidden_states=decoder_out.hidden_states,
            cross_attentions=decoder_out.cross_attentions,
        )
        return decoder_out, loss

    def forward(
        self,
        batch: dict[str, Any],
        return_stage2_attentions: bool = True,
        return_decoder_cross_attentions: bool = False,
    ) -> ProductFactorModelOutput:
        qformer_memory, stage2_distill_attentions = self.encode_qformer(
            batch,
            return_stage1_attentions=False,
            return_stage2_attentions=return_stage2_attentions,
        )
        if self.training and self.use_permutation_ce:
            decoder_out, ce_loss = self._permutation_ce_loss(
                qformer_memory=qformer_memory,
                batch=batch,
                return_decoder_cross_attentions=return_decoder_cross_attentions,
            )
        else:
            decoder_batch = self.decoder_tokenizer.encode_batch_from_dataset(
                batch,
                device=self.device,
            )
            decoder_out = self.decoder(
                input_ids=decoder_batch.input_ids,
                encoder_memory=qformer_memory,
                attention_mask=decoder_batch.attention_mask,
                labels=decoder_batch.labels,
                encoder_attention_mask=None,
                return_cross_attentions=return_decoder_cross_attentions,
            )
            ce_loss = decoder_out.loss
        return ProductFactorModelOutput(
            logits=decoder_out.logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            qformer_memory=qformer_memory,
            stage2_distill_attentions=stage2_distill_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Any],
        max_new_tokens: int = 96,
        return_stage2_attentions: bool = False,
    ) -> ProductFactorModelOutput:
        self.eval()
        qformer_memory, stage2_distill_attentions = self.encode_qformer(
            batch,
            return_stage1_attentions=False,
            return_stage2_attentions=return_stage2_attentions,
        )
        generated_ids = self.decoder.generate(
            encoder_memory=qformer_memory,
            max_new_tokens=max_new_tokens,
            bos_token_id=self.decoder_tokenizer.bos_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
        )
        generated_text = self.decoder_tokenizer.decode_batch(generated_ids)
        return ProductFactorModelOutput(
            logits=torch.empty(0, device=self.device),
            loss=None,
            qformer_memory=qformer_memory,
            stage2_distill_attentions=stage2_distill_attentions,
            generated_ids=generated_ids,
            generated_text=generated_text,
        )


def build_product_factor_model(**kwargs: Any) -> ProductFactorModel:
    """Factory for training scripts."""
    return ProductFactorModel(**kwargs)
