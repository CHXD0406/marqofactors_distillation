#!/usr/bin/env python3
"""Dataset utilities for multimodal product factor generation.

Expected data layout:

    Attribution/dataset/
      train.json
      test.json
      train/{index}.jpg
      test/{index}.jpg
      train_map/{index}.pt
      test_map/{index}.pt

Each JSON record should contain at least `index`, `title`, `image`, and
`pkuseg_factors`. `factors` is used as the autoregressive decoder target when
available. Qwen selected-head maps are loaded from `{split}_map/{index}.pt` and
resized in the collate function to match the DINO 518px grid, i.e. 37x37 for
patch size 14.

The dataset deliberately does not load DINO/RoBERTa/decoder models. It only
loads images, raw text fields, and compact teacher maps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

SplitName = Literal["train", "test"]

DEFAULT_DATASET_ROOT = Path("/mnt/disk5/syh_flowwif2/Attribution/dataset")
DINO_IMAGE_SIZE = 518
DINO_PATCH_SIZE = 14
DINO_GRID_SIZE = DINO_IMAGE_SIZE // DINO_PATCH_SIZE  # 37
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_factor_records(json_path: str | Path) -> list[dict[str, Any]]:
    """Load records from a train/test JSON file."""
    path = Path(json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported dataset JSON structure: {path}")

    if not isinstance(records, list):
        raise ValueError(f"records must be a list in {path}")
    return [record for record in records if isinstance(record, dict)]


def clean_factor_list(value: Any) -> list[str]:
    """Normalize factors only enough for training safety, without changing content."""
    if not isinstance(value, list):
        return []
    factors: list[str] = []
    seen: set[str] = set()
    for item in value:
        factor = str(item or "").strip()
        if not factor:
            continue
        key = factor.lower()
        if key in seen:
            continue
        seen.add(key)
        factors.append(factor)
    return factors


def factors_to_text(factors: list[str], output_format: str = "json_array") -> str:
    """Convert factor list to the text target consumed by the decoder."""
    if output_format == "json_array":
        return json.dumps(factors, ensure_ascii=False, separators=(",", ":"))
    if output_format == "newline":
        return "\n".join(factors)
    if output_format == "semicolon":
        return "; ".join(factors)
    raise ValueError(f"Unsupported output_format: {output_format}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _resolve_image_path(record: dict[str, Any], image_root: Path) -> Path:
    """Resolve image path from record fields, falling back to index-based names."""
    image_value = str(record.get("image", "") or "").strip()
    if image_value:
        image_path = image_root / image_value
        if image_path.exists():
            return image_path

    index = _safe_int(record.get("index"), -1)
    if index >= 0:
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            image_path = image_root / f"{index}{suffix}"
            if image_path.exists():
                return image_path

    if image_value:
        return image_root / image_value
    return image_root / f"{index}.jpg"


def _resolve_map_path(record: dict[str, Any], map_root: Path) -> Path:
    index = _safe_int(record.get("index"), -1)
    return map_root / f"{index}.pt"


def letterbox_resize_to_tensor(
    image: Image.Image,
    image_size: int = DINO_IMAGE_SIZE,
    fill: tuple[int, int, int] = (255, 255, 255),
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resize with aspect ratio preserved, pad to square, and return CHW tensor.

    The returned tensor is ImageNet-normalized by default for DINOv2. Metadata
    includes the original size, resized content size, and content box in the
    padded square, which is useful for later map/box alignment if needed.
    """
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    scale = min(float(image_size) / max(orig_w, 1), float(image_size) / max(orig_h, 1))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    pad_left = (image_size - new_w) // 2
    pad_top = (image_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    meta = {
        "original_size": [orig_w, orig_h],
        "resized_size": [new_w, new_h],
        "image_size": image_size,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "content_box_xyxy": [pad_left, pad_top, pad_left + new_w, pad_top + new_h],
    }
    return tensor, meta


class ProductFactorMapDataset(Dataset):
    """Split-aware dataset for product factor generation with Qwen map targets.

    Only `dataset_root` and `split` are needed. Paths are inferred as:
    - `{dataset_root}/{split}.json`
    - `{dataset_root}/{split}` for images
    - `{dataset_root}/{split}_map` for teacher maps

    Samples are kept only when required fields, image file, and map file exist.
    """

    def __init__(
        self,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        split: SplitName = "train",
        image_size: int = DINO_IMAGE_SIZE,
        output_format: str = "json_array",
        require_pkuseg_factors: bool = True,
        require_target_factors: bool = True,
        require_map: bool = True,
        max_pkuseg_factors: int | None = None,
        normalize_image: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split: SplitName = split
        self.json_path = self.dataset_root / f"{split}.json"
        self.image_root = self.dataset_root / split
        self.map_root = self.dataset_root / f"{split}_map"
        self.image_size = int(image_size)
        self.output_format = output_format
        self.require_pkuseg_factors = bool(require_pkuseg_factors)
        self.require_target_factors = bool(require_target_factors)
        self.require_map = bool(require_map)
        self.max_pkuseg_factors = max_pkuseg_factors
        self.normalize_image = bool(normalize_image)

        records = load_factor_records(self.json_path)
        self.records: list[dict[str, Any]] = []
        self.image_paths: list[Path] = []
        self.map_paths: list[Path] = []
        self.filtered_counts = {
            "missing_pkuseg_factors": 0,
            "missing_target_factors": 0,
            "missing_image": 0,
            "missing_map": 0,
        }

        for record in records:
            pkuseg_factors = clean_factor_list(record.get("pkuseg_factors"))
            target_factors = clean_factor_list(record.get("factors"))
            if self.require_pkuseg_factors and not pkuseg_factors:
                self.filtered_counts["missing_pkuseg_factors"] += 1
                continue
            if self.require_target_factors and not target_factors:
                self.filtered_counts["missing_target_factors"] += 1
                continue

            image_path = _resolve_image_path(record, self.image_root)
            if not image_path.exists():
                self.filtered_counts["missing_image"] += 1
                continue

            map_path = _resolve_map_path(record, self.map_root)
            if self.require_map and not map_path.exists():
                self.filtered_counts["missing_map"] += 1
                continue

            self.records.append(record)
            self.image_paths.append(image_path)
            self.map_paths.append(map_path)

        if not self.records:
            raise ValueError(
                f"No usable records loaded from {self.json_path}; "
                f"filtered_counts={self.filtered_counts}"
            )
        print(
            f"Loaded {split} dataset: {len(self.records)} samples "
            f"from {self.json_path} | filtered={self.filtered_counts}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_path = self.image_paths[idx]
        map_path = self.map_paths[idx]
        index = _safe_int(record.get("index"), idx)

        image = Image.open(image_path).convert("RGB")
        image_tensor, image_meta = letterbox_resize_to_tensor(
            image,
            image_size=self.image_size,
            normalize=self.normalize_image,
        )

        pkuseg_factors = clean_factor_list(record.get("pkuseg_factors"))
        if self.max_pkuseg_factors is not None:
            pkuseg_factors = pkuseg_factors[: self.max_pkuseg_factors]
        target_factors = clean_factor_list(record.get("factors"))
        target_text = factors_to_text(target_factors, output_format=self.output_format)

        map_obj = torch.load(map_path, map_location="cpu") if map_path.exists() else None
        if map_obj is None:
            head_saliency = torch.empty(0)
            mean_saliency = torch.empty(0)
            map_grid_h = 0
            map_grid_w = 0
            selected_heads: list[dict[str, int]] = []
            map_original_size = image_meta["original_size"]
            map_input_size = image_meta["original_size"]
        else:
            head_saliency = map_obj["head_saliency"].float()
            mean_saliency = map_obj["mean_saliency"].float()
            map_grid_h = int(map_obj.get("grid_h", mean_saliency.shape[-2]))
            map_grid_w = int(map_obj.get("grid_w", mean_saliency.shape[-1]))
            selected_heads = list(map_obj.get("selected_heads", []))
            map_original_size = list(map_obj.get("original_image_size", image_meta["original_size"]))
            map_input_size = list(map_obj.get("input_image_size", map_original_size))

        return {
            "index": index,
            "split": self.split,
            "product_id": str(record.get("product_id", record.get("index", idx))),
            "title": str(record.get("title", "") or ""),
            "pkuseg_factors": pkuseg_factors,
            "image": image_tensor,
            "image_path": str(image_path),
            "image_meta": image_meta,
            "map_path": str(map_path),
            "head_saliency": head_saliency,
            "mean_saliency": mean_saliency,
            "map_grid_h": map_grid_h,
            "map_grid_w": map_grid_w,
            "map_original_size": map_original_size,
            "map_input_size": map_input_size,
            "selected_heads": selected_heads,
            "target_factors": target_factors,
            "target_text": target_text,
            "keyword": str(record.get("keyword", "") or ""),
            "brand": str(record.get("brand", "") or ""),
        }


# Backward-compatible alias for earlier imports.
ProductFactorGenerationDataset = ProductFactorMapDataset


def _resize_teacher_maps_direct(
    maps: list[torch.Tensor],
    size: tuple[int, int] = (DINO_GRID_SIZE, DINO_GRID_SIZE),
) -> torch.Tensor:
    resized = []
    for m in maps:
        if m.ndim == 2:
            x = m.unsqueeze(0).unsqueeze(0)
            y = F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        elif m.ndim == 3:
            x = m.unsqueeze(0)
            y = F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).squeeze(0)
        else:
            raise ValueError(f"Unsupported teacher map shape: {tuple(m.shape)}")
        resized.append(y.contiguous())
    return torch.stack(resized, dim=0)


def _resize_single_teacher_map_to_dino_letterbox(
    teacher_map: torch.Tensor,
    image_meta: dict[str, Any],
    output_grid: tuple[int, int] = (DINO_GRID_SIZE, DINO_GRID_SIZE),
) -> torch.Tensor:
    if teacher_map.numel() == 0:
        return teacher_map
    if teacher_map.ndim not in {2, 3}:
        raise ValueError(f"Unsupported teacher map shape: {tuple(teacher_map.shape)}")

    out_h, out_w = output_grid
    content_x0, content_y0, content_x1, content_y1 = image_meta["content_box_xyxy"]
    image_size = int(image_meta["image_size"])
    gx0 = int(np.floor(content_x0 / image_size * out_w))
    gy0 = int(np.floor(content_y0 / image_size * out_h))
    gx1 = int(np.ceil(content_x1 / image_size * out_w))
    gy1 = int(np.ceil(content_y1 / image_size * out_h))
    gx0 = max(0, min(out_w - 1, gx0))
    gy0 = max(0, min(out_h - 1, gy0))
    gx1 = max(gx0 + 1, min(out_w, gx1))
    gy1 = max(gy0 + 1, min(out_h, gy1))
    content_grid_h = gy1 - gy0
    content_grid_w = gx1 - gx0

    if teacher_map.ndim == 2:
        x = teacher_map.unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            x,
            size=(content_grid_h, content_grid_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        aligned = teacher_map.new_zeros((out_h, out_w), dtype=torch.float32)
        aligned[gy0:gy1, gx0:gx1] = resized
    else:
        x = teacher_map.unsqueeze(0).float()
        resized = F.interpolate(
            x,
            size=(content_grid_h, content_grid_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        aligned = teacher_map.new_zeros((teacher_map.shape[0], out_h, out_w), dtype=torch.float32)
        aligned[:, gy0:gy1, gx0:gx1] = resized
    return aligned.contiguous()


def _resize_teacher_maps_to_dino_letterbox(
    maps: list[torch.Tensor],
    image_metas: list[dict[str, Any]],
    size: tuple[int, int] = (DINO_GRID_SIZE, DINO_GRID_SIZE),
) -> torch.Tensor:
    if len(maps) != len(image_metas):
        raise ValueError(f"maps/image_metas length mismatch: {len(maps)} vs {len(image_metas)}")
    return torch.stack(
        [
            _resize_single_teacher_map_to_dino_letterbox(m, meta, output_grid=size)
            for m, meta in zip(maps, image_metas)
        ],
        dim=0,
    )


def product_factor_generation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fields and resize teacher maps to the DINO 518px token grid."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    image_metas = [item["image_meta"] for item in batch]
    head_maps = _resize_teacher_maps_to_dino_letterbox(
        [item["head_saliency"] for item in batch],
        image_metas,
    )
    mean_maps = _resize_teacher_maps_to_dino_letterbox(
        [item["mean_saliency"] for item in batch],
        image_metas,
    )

    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "split": [item["split"] for item in batch],
        "product_id": [item["product_id"] for item in batch],
        "title": [item["title"] for item in batch],
        "pkuseg_factors": [item["pkuseg_factors"] for item in batch],
        "image": images,
        "image_path": [item["image_path"] for item in batch],
        "image_meta": image_metas,
        "map_path": [item["map_path"] for item in batch],
        "head_saliency": head_maps,
        "mean_saliency": mean_maps,
        "original_map_grid": torch.tensor(
            [[item["map_grid_h"], item["map_grid_w"]] for item in batch],
            dtype=torch.long,
        ),
        "selected_heads": [item["selected_heads"] for item in batch],
        "target_factors": [item["target_factors"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "keyword": [item["keyword"] for item in batch],
        "brand": [item["brand"] for item in batch],
    }


def build_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split: SplitName = "train",
    image_size: int = DINO_IMAGE_SIZE,
    output_format: str = "json_array",
    require_pkuseg_factors: bool = True,
    require_target_factors: bool = True,
    require_map: bool = True,
    max_pkuseg_factors: int | None = None,
    normalize_image: bool = True,
) -> ProductFactorMapDataset:
    return ProductFactorMapDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        output_format=output_format,
        require_pkuseg_factors=require_pkuseg_factors,
        require_target_factors=require_target_factors,
        require_map=require_map,
        max_pkuseg_factors=max_pkuseg_factors,
        normalize_image=normalize_image,
    )


def build_train_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    image_size: int = DINO_IMAGE_SIZE,
    output_format: str = "json_array",
    require_pkuseg_factors: bool = True,
    require_target_factors: bool = True,
    require_map: bool = True,
    max_pkuseg_factors: int | None = None,
    normalize_image: bool = True,
) -> ProductFactorMapDataset:
    return build_dataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        output_format=output_format,
        require_pkuseg_factors=require_pkuseg_factors,
        require_target_factors=require_target_factors,
        require_map=require_map,
        max_pkuseg_factors=max_pkuseg_factors,
        normalize_image=normalize_image,
    )


def build_test_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    image_size: int = DINO_IMAGE_SIZE,
    output_format: str = "json_array",
    require_pkuseg_factors: bool = True,
    require_target_factors: bool = True,
    require_map: bool = True,
    max_pkuseg_factors: int | None = None,
    normalize_image: bool = True,
) -> ProductFactorMapDataset:
    return build_dataset(
        dataset_root=dataset_root,
        split="test",
        image_size=image_size,
        output_format=output_format,
        require_pkuseg_factors=require_pkuseg_factors,
        require_target_factors=require_target_factors,
        require_map=require_map,
        max_pkuseg_factors=max_pkuseg_factors,
        normalize_image=normalize_image,
    )


def build_train_test_datasets(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    image_size: int = DINO_IMAGE_SIZE,
    output_format: str = "json_array",
    require_pkuseg_factors: bool = True,
    require_target_factors: bool = True,
    require_map: bool = True,
    max_pkuseg_factors: int | None = None,
    normalize_image: bool = True,
) -> tuple[ProductFactorMapDataset, ProductFactorMapDataset]:
    """Explicitly build separated train/test datasets for validation/early stop."""
    train_dataset = build_train_dataset(
        dataset_root=dataset_root,
        image_size=image_size,
        output_format=output_format,
        require_pkuseg_factors=require_pkuseg_factors,
        require_target_factors=require_target_factors,
        require_map=require_map,
        max_pkuseg_factors=max_pkuseg_factors,
        normalize_image=normalize_image,
    )
    test_dataset = build_test_dataset(
        dataset_root=dataset_root,
        image_size=image_size,
        output_format=output_format,
        require_pkuseg_factors=require_pkuseg_factors,
        require_target_factors=require_target_factors,
        require_map=require_map,
        max_pkuseg_factors=max_pkuseg_factors,
        normalize_image=normalize_image,
    )
    return train_dataset, test_dataset


def dataset_summary(dataset: ProductFactorMapDataset) -> dict[str, Any]:
    pkuseg_counts = [len(clean_factor_list(record.get("pkuseg_factors"))) for record in dataset.records]
    target_counts = [len(clean_factor_list(record.get("factors"))) for record in dataset.records]
    missing_images = sum(1 for image_path in dataset.image_paths if not image_path.exists())
    missing_maps = sum(1 for map_path in dataset.map_paths if not map_path.exists())
    return {
        "dataset_root": str(dataset.dataset_root),
        "split": dataset.split,
        "json_path": str(dataset.json_path),
        "image_root": str(dataset.image_root),
        "map_root": str(dataset.map_root),
        "records": len(dataset),
        "missing_images": missing_images,
        "missing_maps": missing_maps,
        "filtered_counts": dict(dataset.filtered_counts),
        "avg_pkuseg_factor_count": sum(pkuseg_counts) / len(pkuseg_counts) if pkuseg_counts else 0.0,
        "max_pkuseg_factor_count": max(pkuseg_counts) if pkuseg_counts else 0,
        "avg_target_factor_count": sum(target_counts) / len(target_counts) if target_counts else 0.0,
        "max_target_factor_count": max(target_counts) if target_counts else 0,
        "max_pkuseg_factors": dataset.max_pkuseg_factors,
        "image_size": dataset.image_size,
        "dino_grid_size": DINO_GRID_SIZE,
        "output_format": dataset.output_format,
    }
