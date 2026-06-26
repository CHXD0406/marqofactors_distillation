#!/usr/bin/env python3
"""Train EN-QFormer-DE product factor model.

Launch examples:

Single GPU:
    python train.py

DDP with selected GPUs:
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train.py

This script keeps all hard-coded project paths in `Config`, supports early
stopping on test loss, best/latest checkpointing, and resume training. In DDP,
every rank owns one full copy of frozen DINO + frozen RoBERTa + trainable
Q-Former/decoder on its local GPU.
"""

from __future__ import annotations

import datetime
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from .dataset import build_test_dataset, build_train_dataset, product_factor_generation_collate_fn
    from .model import ProductFactorModel
except ImportError:
    from dataset import build_test_dataset, build_train_dataset, product_factor_generation_collate_fn  # type: ignore
    from model import ProductFactorModel  # type: ignore


@dataclass
class Config:
    # -------- paths --------
    dataset_root: str = "/mnt/disk5/syh_flowwif2/Attribution/dataset"
    ckpt_dir: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/checkpoints"
    text_model_path: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/chinese-roberta-wwm-ext"
    vision_model_path: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/dinov2-large"
    decoder_tokenizer_path: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/decoder_tokenizer"
    decoder_embedding_init_path: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/chinese-roberta-wwm-ext"
    resume_ckpt: str | None = None

    # -------- data --------
    image_size: int = 518
    max_pkuseg_factors: int = 24
    num_workers: int = 10
    require_map: bool = True
    require_target_factors: bool = True

    # -------- optimization --------
    num_epochs: int = 100
    batch_size: int = 25
    accumulation_steps: int = 12
    lr: float = 8e-5
    min_lr: float = 8e-6
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    patience: int = 10
    seed: int = 42

    # -------- loss weights --------
    ce_weight: float = 1.0
    map_weight: float = 0.1

    # -------- AMP/DDP --------
    prefer_bf16: bool = True
    encoder_dtype: str = "bf16"  # bf16/fp16/fp32 for frozen encoders
    find_unused_parameters: bool = False

    # -------- model --------
    text_max_length: int = 256
    decoder_max_length: int = 128
    init_decoder_from_roberta: bool = True
    use_permutation_ce: bool = True
    perm_ce_max_samples: int = 4
    perm_ce_tau: float = 0.2

    # -------- logging/checkpoint --------
    log_every: int = 2
    save_latest_every_epoch: bool = True
    eval_dir: str = "/mnt/disk5/syh_flowwif2/Attribution/model_factors/eval"
    eval_sample_count: int = 50
    eval_max_new_tokens: int = 96


CFG = Config()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def ddp_setup() -> tuple[torch.device, int, int, bool]:
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=6))
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device, local_rank, world_size, is_distributed


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_dtype() -> torch.dtype | None:
    if not torch.cuda.is_available():
        return None
    if CFG.prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def encoder_torch_dtype() -> torch.dtype | None:
    value = CFG.encoder_dtype.lower().strip()
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return None
    raise ValueError(f"Unsupported encoder_dtype: {CFG.encoder_dtype}")


def reduce_mean(value: torch.Tensor, is_distributed: bool) -> torch.Tensor:
    value = value.detach().clone()
    if is_distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float().clamp_min(eps)
    denom = x.flatten(-2).sum(dim=-1).clamp_min(eps)
    return x / denom[..., None, None]


def qwen_map_distill_loss(
    stage2_attentions: dict[int, torch.Tensor],
    teacher_maps: torch.Tensor,
    image_grid_h: int = 37,
    image_grid_w: int = 37,
) -> torch.Tensor:
    """KL teacher Qwen maps against selected Stage2 image cross-attn heads.

    stage2_attentions[layer]: [B,10,40,N]
    teacher_maps: [B,10,37,37]

    We align selected student heads 0..9 to teacher maps 0..9 and average over
    query tokens, so each selected head yields one image saliency map.
    """
    if not stage2_attentions:
        return teacher_maps.new_tensor(0.0)

    teacher = normalize_map(teacher_maps)
    losses = []
    for attn in stage2_attentions.values():
        bsz, num_heads, _, num_image_tokens = attn.shape
        if num_image_tokens != image_grid_h * image_grid_w:
            raise ValueError(f"attention N={num_image_tokens} does not match grid {image_grid_h}x{image_grid_w}")
        student = attn.float().mean(dim=2).view(bsz, num_heads, image_grid_h, image_grid_w)
        student = normalize_map(student)
        target = teacher[:, :num_heads]
        kl = F.kl_div(student.clamp_min(1e-6).log(), target, reduction="batchmean")
        losses.append(kl)
    return torch.stack(losses).mean()


def build_loaders(is_distributed: bool) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    train_dataset = build_train_dataset(
        dataset_root=CFG.dataset_root,
        image_size=CFG.image_size,
        require_target_factors=CFG.require_target_factors,
        require_map=CFG.require_map,
        max_pkuseg_factors=CFG.max_pkuseg_factors,
    )
    test_dataset = build_test_dataset(
        dataset_root=CFG.dataset_root,
        image_size=CFG.image_size,
        require_target_factors=CFG.require_target_factors,
        require_map=CFG.require_map,
        max_pkuseg_factors=CFG.max_pkuseg_factors,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CFG.num_workers > 0,
        collate_fn=product_factor_generation_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CFG.num_workers > 0,
        collate_fn=product_factor_generation_collate_fn,
    )
    return train_loader, test_loader, train_sampler, test_sampler


def build_model(device: torch.device) -> ProductFactorModel:
    model = ProductFactorModel(
        text_model_path=CFG.text_model_path,
        vision_model_path=CFG.vision_model_path,
        tokenizer_path=CFG.decoder_tokenizer_path,
        roberta_path_for_embedding_init=CFG.decoder_embedding_init_path,
        text_max_length=CFG.text_max_length,
        decoder_max_length=CFG.decoder_max_length,
        device=device,
        encoder_dtype=encoder_torch_dtype(),
        init_decoder_from_roberta=CFG.init_decoder_from_roberta,
        use_permutation_ce=CFG.use_permutation_ce,
        perm_ce_max_samples=CFG.perm_ce_max_samples,
        perm_ce_tau=CFG.perm_ce_tau,
    )
    return model


def make_optimizer(model_raw: ProductFactorModel) -> torch.optim.Optimizer:
    trainable = [p for p in model_raw.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=CFG.lr,
        betas=CFG.betas,
        weight_decay=CFG.weight_decay,
    )


def trainable_state_dict(model_raw: ProductFactorModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model_raw.state_dict().items() if not k.startswith("encoders.")}


def load_trainable_state_dict(model_raw: ProductFactorModel, state_dict: dict[str, torch.Tensor]) -> Any:
    current = model_raw.state_dict()
    merged = dict(current)
    for key, value in state_dict.items():
        if key in merged and merged[key].shape == value.shape:
            merged[key] = value
    return model_raw.load_state_dict(merged, strict=False)


def save_checkpoint(
    path: Path,
    model_raw: ProductFactorModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_loss: float,
    patience_counter: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": trainable_state_dict(model_raw),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "config": asdict(CFG),
        },
        path,
    )


def resume_if_needed(
    model_raw: ProductFactorModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not CFG.resume_ckpt:
        return 0, float("inf"), 0
    ckpt_path = Path(CFG.resume_ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=device)
    load_trainable_state_dict(model_raw, ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_loss = float(ckpt.get("best_loss", float("inf")))
    patience_counter = int(ckpt.get("patience_counter", 0))
    if is_main_process():
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss, patience_counter


def compute_loss(model_out: Any, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if model_out.loss is None:
        raise RuntimeError("decoder CE loss is None during training")
    ce_loss = model_out.loss
    teacher_maps = batch["head_saliency"].to(device=device, non_blocking=True)
    map_loss = qwen_map_distill_loss(model_out.stage2_distill_attentions, teacher_maps)
    total = CFG.ce_weight * ce_loss + CFG.map_weight * map_loss
    weighted_ce = CFG.ce_weight * ce_loss
    weighted_map = CFG.map_weight * map_loss
    return total, {
        "ce": weighted_ce.detach(),
        "map": weighted_map.detach(),
        "raw_ce": ce_loss.detach(),
        "raw_map": map_loss.detach(),
        "total": total.detach(),
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    is_distributed: bool,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dtype = amp_dtype()
    use_amp = dtype is not None
    running = {"total": 0.0, "ce": 0.0, "map": 0.0, "raw_ce": 0.0, "raw_map": 0.0}
    num_steps = 0
    dataset_size = len(loader.dataset)
    global_batch_size = CFG.batch_size * (dist.get_world_size() if is_distributed else 1)
    progress = tqdm(
        total=dataset_size,
        disable=not is_main_process(),
        desc=f"Epoch {epoch:03d}",
        unit="img",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for step, batch in enumerate(loader):
        with autocast(enabled=use_amp, dtype=dtype):
            out = model(batch, return_stage2_attentions=True, return_decoder_cross_attentions=False)
            loss, parts = compute_loss(out, batch, device)
            loss = loss / CFG.accumulation_steps

        if use_amp and dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        do_step = (step + 1) % CFG.accumulation_steps == 0 or (step + 1) == len(loader)
        if do_step:
            if use_amp and dtype == torch.float16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            if use_amp and dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key in running:
            running[key] += float(parts[key].item())
        num_steps += 1

        if is_main_process():
            progress.update(min(global_batch_size, dataset_size - progress.n))
            if (step + 1) % CFG.log_every == 0 or (step + 1) == len(loader):
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    total=f"{running['total']/num_steps:.4f}",
                    ce_w=f"{running['ce']/num_steps:.4f}",
                    map_w=f"{running['map']/num_steps:.4f}",
                    lr=f"{lr:.2e}",
                )

    if is_main_process():
        progress.close()

    metrics = {k: torch.tensor(v / max(num_steps, 1), device=device) for k, v in running.items()}
    metrics = {k: float(reduce_mean(v, is_distributed).item()) for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_distributed: bool,
) -> dict[str, float]:
    model.eval()
    dtype = amp_dtype()
    use_amp = dtype is not None
    running = {"total": 0.0, "ce": 0.0, "map": 0.0}
    num_steps = 0
    for batch in loader:
        with autocast(enabled=use_amp, dtype=dtype):
            out = model(batch, return_stage2_attentions=True, return_decoder_cross_attentions=False)
            _, parts = compute_loss(out, batch, device)
        for key in running:
            running[key] += float(parts[key].item())
        num_steps += 1

    metrics = {k: torch.tensor(v / max(num_steps, 1), device=device) for k, v in running.items()}
    metrics = {k: float(reduce_mean(v, is_distributed).item()) for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def save_epoch_eval_samples(
    model_raw: ProductFactorModel,
    test_dataset: Any,
    epoch: int,
    device: torch.device,
) -> None:
    if CFG.eval_sample_count <= 0:
        return
    eval_dir = Path(CFG.eval_dir).expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(CFG.eval_sample_count), len(test_dataset))
    rng = random.Random(CFG.seed + epoch * 10007)
    indices = rng.sample(range(len(test_dataset)), sample_count)

    records: list[dict[str, Any]] = []
    was_training = model_raw.training
    model_raw.eval()
    for idx in indices:
        sample = test_dataset[idx]
        batch = product_factor_generation_collate_fn([sample])
        out = model_raw.generate(
            batch,
            max_new_tokens=CFG.eval_max_new_tokens,
            return_stage2_attentions=False,
        )
        prediction = out.generated_text[0] if out.generated_text else ""
        records.append(
            {
                "index": sample.get("index"),
                "title": sample.get("title"),
                "correct_answer": sample.get("target_text"),
                "predicted_answer": prediction,
                "正确答案-预测答案": f"{sample.get('target_text')} - {prediction}",
            }
        )
    if was_training:
        model_raw.train()

    output_path = eval_dir / f"epoch_{epoch:03d}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved eval samples: {output_path}")


def main() -> None:
    device, local_rank, world_size, is_distributed = ddp_setup()
    set_seed(CFG.seed, local_rank)
    ckpt_dir = Path(CFG.ckpt_dir).expanduser().resolve()

    if is_main_process():
        print(f"Device: {device} | world_size={world_size}")
        print(f"Checkpoints: {ckpt_dir}")
        print(f"Config: {asdict(CFG)}")

    train_loader, test_loader, train_sampler, _ = build_loaders(is_distributed)
    model_raw = build_model(device)
    model: nn.Module = model_raw
    if is_distributed:
        model = DDP(
            model_raw,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=CFG.find_unused_parameters,
        )

    optimizer = make_optimizer(model_raw)
    total_update_steps = max(1, (len(train_loader) // max(1, CFG.accumulation_steps)) * CFG.num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_update_steps,
        eta_min=CFG.min_lr,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available() and amp_dtype() == torch.float16)
    start_epoch, best_loss, patience_counter = resume_if_needed(model_raw, optimizer, scheduler, scaler, device)

    try:
        for epoch in range(start_epoch, CFG.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(model, optimizer, scheduler, scaler, train_loader, device, epoch, is_distributed)
            test_metrics = evaluate(model, test_loader, device, is_distributed)

            if is_main_process():
                save_epoch_eval_samples(model_raw, test_loader.dataset, epoch, device)
                print(
                    f"Epoch {epoch:03d} done | "
                    f"train {train_metrics} | test {test_metrics} | best {best_loss:.6f}"
                )

                improved = test_metrics["total"] < best_loss
                if improved:
                    best_loss = test_metrics["total"]
                    patience_counter = 0
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model_raw,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_loss,
                        patience_counter,
                    )
                    print(f"Saved best checkpoint: epoch={epoch}, test_total={best_loss:.6f}")
                else:
                    patience_counter += 1

                if CFG.save_latest_every_epoch:
                    save_checkpoint(
                        ckpt_dir / "latest.pt",
                        model_raw,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_loss,
                        patience_counter,
                    )

                should_stop = patience_counter >= CFG.patience
            else:
                should_stop = False

            if is_distributed:
                flag = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int32)
                dist.broadcast(flag, src=0)
                should_stop = bool(flag.item())

            if should_stop:
                if is_main_process():
                    print(f"Early stopping at epoch {epoch}; patience={CFG.patience}")
                break
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
