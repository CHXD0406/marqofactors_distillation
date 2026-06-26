#!/usr/bin/env python3
"""Frozen encoder wrappers for product factor modeling.

This module connects:
- DINOv2 image encoder: image tensor [B,3,518,518] -> patch tokens
- Chinese RoBERTa encoder: title + noisy pkuseg factors -> full text tokens

The dataset already performs letterbox resize/pad and ImageNet normalization for
DINOv2, so `encode_image` consumes tensors directly and does not call an image
processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, Dinov2Model

DEFAULT_MODEL_ROOT = Path("/mnt/disk5/syh_flowwif2/Attribution/model_factors")
DEFAULT_TEXT_MODEL = DEFAULT_MODEL_ROOT / "chinese-roberta-wwm-ext"
DEFAULT_VISION_MODEL = DEFAULT_MODEL_ROOT / "dinov2-large"


@dataclass
class TextEncoderOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    last_hidden_state: torch.Tensor
    cls: torch.Tensor


@dataclass
class VisionEncoderOutput:
    last_hidden_state: torch.Tensor
    patch_tokens: torch.Tensor
    cls: torch.Tensor
    grid_h: int
    grid_w: int


def _weight_files(path: Path) -> list[Path]:
    return [p for p in (path / "model.safetensors", path / "pytorch_model.bin") if p.exists() and p.stat().st_size > 0]


def resolve_device(device: torch.device | str | int | None = None) -> torch.device:
    """Resolve device in a DDP-friendly way.

    Priority:
    1. Explicit `device` argument.
    2. `LOCAL_RANK` from torchrun, mapped to `cuda:{LOCAL_RANK}`.
    3. `cuda` if available, otherwise CPU.

    This lets every DDP process instantiate its own frozen encoders and student
    replica on the correct GPU without hard-coding cuda:0.
    """
    if device is not None:
        if isinstance(device, torch.device):
            resolved = device
        elif isinstance(device, int):
            resolved = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(str(device))
    elif torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        resolved = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    elif torch.cuda.is_available():
        resolved = torch.device("cuda")
    else:
        resolved = torch.device("cpu")

    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    return resolved


def assert_local_model_ready(path: str | Path, model_name: str) -> None:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{model_name} directory does not exist: {path}")
    if not _weight_files(path):
        raise FileNotFoundError(
            f"{model_name} usable weights are missing in {path}. Expected non-empty model.safetensors or pytorch_model.bin."
        )


class FrozenTextEncoder(nn.Module):
    """Frozen Chinese RoBERTa/BERT encoder returning full token embeddings."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TEXT_MODEL,
        max_length: int = 256,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        assert_local_model_ready(self.model_path, "Chinese RoBERTa")
        self.max_length = int(max_length)
        self._device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.encoder = AutoModel.from_pretrained(str(self.model_path), torch_dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(self._device)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self._device

    def build_text_pair(self, titles: list[str], pkuseg_factors: list[list[str]]) -> tuple[list[str], list[str]]:
        text_a = ["标题：" + str(title or "") for title in titles]
        text_b = [
            "强噪声参考：" + "；".join(str(f or "").strip() for f in factors if str(f or "").strip())
            for factors in pkuseg_factors
        ]
        return text_a, text_b

    @torch.no_grad()
    def forward(self, titles: list[str], pkuseg_factors: list[list[str]]) -> TextEncoderOutput:
        text_a, text_b = self.build_text_pair(titles, pkuseg_factors)
        encoded = self.tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded, return_dict=True)
        token_type_ids = encoded.get("token_type_ids")
        return TextEncoderOutput(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=token_type_ids,
            last_hidden_state=outputs.last_hidden_state,
            cls=outputs.last_hidden_state[:, 0],
        )


class FrozenDINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder consuming normalized image tensors directly."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_VISION_MODEL,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        assert_local_model_ready(self.model_path, "DINOv2")
        self._device = resolve_device(device)
        self.encoder = Dinov2Model.from_pretrained(str(self.model_path), torch_dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(self._device)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    @property
    def patch_size(self) -> int:
        return int(self.encoder.config.patch_size)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        images = images.to(device=self.device, dtype=next(self.encoder.parameters()).dtype, non_blocking=True)
        outputs = self.encoder(pixel_values=images, return_dict=True)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]
        patches = hidden[:, 1:]
        _, _, image_h, image_w = images.shape
        grid_h = image_h // self.patch_size
        grid_w = image_w // self.patch_size
        expected = grid_h * grid_w
        if patches.shape[1] != expected:
            raise RuntimeError(
                f"DINO patch token count mismatch: got {patches.shape[1]}, expected {expected} "
                f"from image {image_h}x{image_w} and patch_size={self.patch_size}"
            )
        return VisionEncoderOutput(
            last_hidden_state=hidden,
            patch_tokens=patches,
            cls=cls,
            grid_h=grid_h,
            grid_w=grid_w,
        )


class ProductFactorFrozenEncoders(nn.Module):
    """Convenience wrapper holding both frozen text and vision encoders.

    In DDP, instantiate this inside each process without passing `device`, or
    pass `device=local_rank`. The wrapper resolves `LOCAL_RANK` and places one
    full encoder replica on each process GPU.
    """

    def __init__(
        self,
        text_model_path: str | Path = DEFAULT_TEXT_MODEL,
        vision_model_path: str | Path = DEFAULT_VISION_MODEL,
        text_max_length: int = 256,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._device = resolve_device(device)
        self.text = FrozenTextEncoder(
            model_path=text_model_path,
            max_length=text_max_length,
            device=self._device,
            dtype=dtype,
        )
        self.vision = FrozenDINOv2Encoder(
            model_path=vision_model_path,
            device=self._device,
            dtype=dtype,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        text_out = self.text(batch["title"], batch["pkuseg_factors"])
        vision_out = self.vision(batch["image"])
        return {
            "text": text_out,
            "vision": vision_out,
            "text_tokens": text_out.last_hidden_state,
            "text_attention_mask": text_out.attention_mask,
            "text_cls": text_out.cls,
            "image_tokens": vision_out.patch_tokens,
            "image_cls": vision_out.cls,
            "image_grid_h": vision_out.grid_h,
            "image_grid_w": vision_out.grid_w,
        }
