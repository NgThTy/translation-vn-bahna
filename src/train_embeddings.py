"""
Fine-tune Bahnaric -> Vietnamese projection heads.

Now supports:
- Contrastive InfoNCE with in-batch negatives (symmetric).
- MSE, InfoNCE, or Hybrid losses via --loss_type.
- Per-epoch logging: grad norms, mean seq lengths, truncation rate.
- Safely uses .item() for scalar logging.

Example:
python train_embeddings.py \
  --train_csv ../data/train.csv \
  --src_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --tgt_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --projection_dim 256 \
  --freeze_base \
  --epochs 3 --batch_size 64 --lr 2e-4 \
  --src_max_len 256 --tgt_max_len 256 \
  --loss_type infonce --temperature 0.05
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import sys

from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger("train_embeddings")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BahVnPairsDataset(Dataset):
    """CSV dataset expecting columns 'Bahnaric' and 'Vietnamese' (strings)."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
            raise ValueError("CSV must have 'Bahnaric' and 'Vietnamese' columns")
        df = df.dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
        self.src_texts = df["Bahnaric"].astype(str).tolist()    # source = Bahnaric
        self.tgt_texts = df["Vietnamese"].astype(str).tolist()  # target = Vietnamese

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.tgt_texts[idx]


def collate_fn(
    batch: List[Tuple[str, str]],
    src_tokenizer,
    tgt_tokenizer,
    src_max_len: int,
    tgt_max_len: int,
    device: torch.device,
):
    src_texts, tgt_texts = zip(*batch)
    src_inputs = src_tokenizer(
        list(src_texts), padding=True, truncation=True, max_length=src_max_len, return_tensors="pt"
    )
    tgt_inputs = tgt_tokenizer(
        list(tgt_texts), padding=True, truncation=True, max_length=tgt_max_len, return_tensors="pt"
    )
    for k in src_inputs:
        src_inputs[k] = src_inputs[k].to(device)
    for k in tgt_inputs:
        tgt_inputs[k] = tgt_inputs[k].to(device)
    return src_inputs, tgt_inputs


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FineTuneModel(nn.Module):
    def __init__(self, base_model_name: str, projection_dim: int = 256, freeze_base: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Cannot determine hidden size from config; check model name")

        self.proj = ProjectionHead(hidden_size, out_dim=projection_dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        outputs = self.base(**input_dict, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        mask = input_dict.get("attention_mask", torch.ones(last_hidden.size()[:2], device=last_hidden.device))
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths  # (B, H)
        projected = self.proj(pooled)  # (B, proj_dim)
        return projected


def build_models(src_model_name, tgt_model_name, proj_dim, freeze_base, device):
    src = FineTuneModel(src_model_name, proj_dim, freeze_base).to(device)
    tgt = FineTuneModel(tgt_model_name, proj_dim, freeze_base).to(device)
    return src, tgt


def apply_lora_if_needed(base_model, args):
    if not args.use_lora:
        return base_model

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # encoders used for embeddings
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # Safety pass: LoRA params should already be trainable, others frozen.
    # We still enforce it explicitly:
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return peft_model


def _make_pbar(iterable, desc):
    total = len(iterable)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="batch",
        leave=False,
        ncols=100,
        disable=not sys.stdout.isatty(),
    )


# -------- Losses --------

def mse_loss(src_emb, tgt_emb):
    return F.mse_loss(src_emb, tgt_emb, reduction="mean")


def infonce_loss(src_emb, tgt_emb, temperature: float = 0.07):
    """
    Symmetric InfoNCE with in-batch negatives.
    src_emb: (B, D), tgt_emb: (B, D)
    """
    src = F.normalize(src_emb, p=2, dim=-1)
    tgt = F.normalize(tgt_emb, p=2, dim=-1)
    logits = torch.matmul(src, tgt.t()) / max(1e-6, temperature)  # (B, B)
    labels = torch.arange(src.size(0), device=src.device)

    loss_src = F.cross_entropy(logits, labels)      # src -> tgt
    loss_tgt = F.cross_entropy(logits.t(), labels)  # tgt -> src
    return 0.5 * (loss_src + loss_tgt)


def hybrid_loss(src_emb, tgt_emb, temperature: float, alpha: float):
    """
    alpha * InfoNCE + (1 - alpha) * MSE
    """
    return alpha * infonce_loss(src_emb, tgt_emb, temperature) + (1 - alpha) * mse_loss(src_emb, tgt_emb)


# -------- Training / Eval --------

def _batch_seq_stats(src_inputs, tgt_inputs, src_max_len, tgt_max_len):
    # mean lengths and crude truncation rate (length == max_len implies likely truncation)
    src_len = src_inputs["attention_mask"].sum(dim=1).float()
    tgt_len = tgt_inputs["attention_mask"].sum(dim=1).float()

    src_trunc = (src_len >= float(src_max_len)).float()
    tgt_trunc = (tgt_len >= float(tgt_max_len)).float()

    return {
        "src_mean_len": src_len.mean().item(),
        "tgt_mean_len": tgt_len.mean().item(),
        "src_trunc_rate": src_trunc.mean().item(),
        "tgt_trunc_rate": tgt_trunc.mean().item(),
    }


def _log_grad_norms(model: nn.Module, tag: str):
    norms = []
    for n, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            norms.append(p.grad.norm().detach())
    if norms:
        total = torch.stack(norms).norm().item()
        mean = torch.stack(norms).mean().item()
        LOGGER.info(f"[GradNorm] {tag}: total={total:.4f} mean_param_grad={mean:.6f}")
    else:
        LOGGER.info(f"[GradNorm] {tag}: no grads (check requires_grad / freeze settings)")


def train_one_epoch(
    src_model,
    tgt_model,
    dataloader,
    optimizer,
    device,
    scheduler=None,
    max_grad_norm=None,
    loss_type: str = "infonce",
    temperature: float = 0.07,
    alpha: float = 0.5,  # for hybrid
    src_max_len: int = 256,
    tgt_max_len: int = 256,
):
    src_model.train()
    tgt_model.train()

    total_loss = 0.0
    n_batches = 0

    # running stats for sequence lengths & truncation
    src_len_sum = tgt_len_sum = 0.0
    src_trunc_sum = tgt_trunc_sum = 0.0
    total_samples = 0

    pbar = _make_pbar(dataloader, "Training")
    for step, (src_inputs, tgt_inputs) in enumerate(pbar, 1):
        optimizer.zero_grad(set_to_none=True)

        src_emb = src_model(src_inputs)
        tgt_emb = tgt_model(tgt_inputs)

        if loss_type == "mse":
            loss = mse_loss(src_emb, tgt_emb)
        elif loss_type == "infonce":
            loss = infonce_loss(src_emb, tgt_emb, temperature=temperature)
        elif loss_type == "hybrid":
            loss = hybrid_loss(src_emb, tgt_emb, temperature=temperature, alpha=alpha)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                max_grad_norm,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        # seq stats
        stats = _batch_seq_stats(src_inputs, tgt_inputs, src_max_len, tgt_max_len)
        bsz = src_inputs["input_ids"].size(0)
        src_len_sum += stats["src_mean_len"] * bsz
        tgt_len_sum += stats["tgt_mean_len"] * bsz
        src_trunc_sum += stats["src_trunc_rate"] * bsz
        tgt_trunc_sum += stats["tgt_trunc_rate"] * bsz
        total_samples += bsz

        if step % 50 == 0 or step == len(dataloader):
            pbar.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

    # per-epoch logs
    LOGGER.info(f"Train loss ({loss_type}): { (total_loss / max(1, n_batches)) :.6f}")
    if total_samples > 0:
        LOGGER.info(
            "Train seq stats: "
            f"src_mean_len={src_len_sum/total_samples:.1f} "
            f"tgt_mean_len={tgt_len_sum/total_samples:.1f} "
            f"src_trunc_rate={src_trunc_sum/total_samples:.3f} "
            f"tgt_trunc_rate={tgt_trunc_sum/total_samples:.3f}"
        )

    # grad norms (after epoch)
    _log_grad_norms(src_model.proj, "src_proj")
    _log_grad_norms(tgt_model.proj, "tgt_proj")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    src_model,
    tgt_model,
    dataloader,
    device,
    loss_type: str = "infonce",
    temperature: float = 0.07,
    alpha: float = 0.5,
    src_max_len: int = 256,
    tgt_max_len: int = 256,
):
    src_model.eval()
    tgt_model.eval()

    total_loss = 0.0
    n_batches = 0

    # running stats for sequence lengths & truncation
    src_len_sum = tgt_len_sum = 0.0
    src_trunc_sum = tgt_trunc_sum = 0.0
    total_samples = 0

    pbar = _make_pbar(dataloader, "Evaluating")
    for step, (src_inputs, tgt_inputs) in enumerate(pbar, 1):
        src_emb = src_model(src_inputs)
        tgt_emb = tgt_model(tgt_inputs)

        if loss_type == "mse":
            loss = mse_loss(src_emb, tgt_emb)
        elif loss_type == "infonce":
            loss = infonce_loss(src_emb, tgt_emb, temperature=temperature)
        elif loss_type == "hybrid":
            loss = hybrid_loss(src_emb, tgt_emb, temperature=temperature, alpha=alpha)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        total_loss += loss.item()
        n_batches += 1

        if step % 50 == 0 or step == len(dataloader):
            pbar.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

        # seq stats
        stats = _batch_seq_stats(src_inputs, tgt_inputs, src_max_len, tgt_max_len)
        bsz = src_inputs["input_ids"].size(0)
        src_len_sum += stats["src_mean_len"] * bsz
        tgt_len_sum += stats["tgt_mean_len"] * bsz
        src_trunc_sum += stats["src_trunc_rate"] * bsz
        tgt_trunc_sum += stats["tgt_trunc_rate"] * bsz
        total_samples += bsz

    LOGGER.info(f"Valid loss ({loss_type}): { (total_loss / max(1, n_batches)) :.6f}")
    if total_samples > 0:
        LOGGER.info(
            "Valid seq stats: "
            f"src_mean_len={src_len_sum/total_samples:.1f} "
            f"tgt_mean_len={tgt_len_sum/total_samples:.1f} "
            f"src_trunc_rate={src_trunc_sum/total_samples:.3f} "
            f"tgt_trunc_rate={tgt_trunc_sum/total_samples:.3f}"
        )

    return total_loss / max(1, n_batches)


def save_models(src_model, tgt_model, out_dir: str, save_base: bool = False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Projection heads
    torch.save(src_model.proj.state_dict(), Path(out_dir) / "src_proj.pt")
    torch.save(tgt_model.proj.state_dict(), Path(out_dir) / "tgt_proj.pt")
    
    # LoRA adapters (if any)
    if hasattr(src_model.base, "save_pretrained"):
        (Path(out_dir) / "src_adapters").mkdir(parents=True, exist_ok=True)
        src_model.base.save_pretrained(Path(out_dir) / "src_adapters")
    if hasattr(tgt_model.base, "save_pretrained"):
        (Path(out_dir) / "tgt_adapters").mkdir(parents=True, exist_ok=True)
        tgt_model.base.save_pretrained(Path(out_dir) / "tgt_adapters")

    with open(Path(out_dir) / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"saved_at: {Path(out_dir).absolute()}\n")
    
    if save_base:
        try:
            base_src = getattr(src_model.base, "get_base_model", lambda: src_model.base)()
            base_src.save_pretrained(Path(out_dir) / "src_base")
        except Exception as e:
            LOGGER.warning(f"Could not save src base: {e}")
        try:
            base_tgt = getattr(tgt_model.base, "get_base_model", lambda: tgt_model.base)()
            base_tgt.save_pretrained(Path(out_dir) / "tgt_base")
        except Exception as e:
            LOGGER.warning(f"Could not save tgt base: {e}")

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    LOGGER.info(f"Using device: {device}")

    # Load data
    train_ds = BahVnPairsDataset(args.train_csv)
    valid_ds = BahVnPairsDataset(args.valid_csv) if args.valid_csv else None

    # Tokenizers
    src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        persistent_workers=False if args.num_workers == 0 else True,
        drop_last=False,
        collate_fn=lambda b: collate_fn(
            b, src_tokenizer, tgt_tokenizer, args.src_max_len, args.tgt_max_len, device
        ),
    )
    valid_loader = None
    if valid_ds:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.pin_memory and device.type == "cuda"),
            persistent_workers=False if args.num_workers == 0 else True,
            drop_last=False,
            collate_fn=lambda b: collate_fn(
                b, src_tokenizer, tgt_tokenizer, args.src_max_len, args.tgt_max_len, device
            ),
        )

    # Build models
    src_model, tgt_model = build_models(args.src_model, args.tgt_model, args.projection_dim, args.freeze_base, device)
    
    # Apply LoRA if requested
    src_model.base = apply_lora_if_needed(src_model.base, args)
    tgt_model.base = apply_lora_if_needed(tgt_model.base, args)

    def count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    LOGGER.info(
        f"Trainable params — "
        f"src_base:{count_trainable(src_model.base)} "
        f"tgt_base:{count_trainable(tgt_model.base)} "
        f"src_proj:{count_trainable(src_model.proj)} "
        f"tgt_proj:{count_trainable(tgt_model.proj)}"
    )

    # Collect parameters to optimize
    optim_params = list(src_model.proj.parameters()) + list(tgt_model.proj.parameters())

    # If LoRA is on, add ONLY LoRA adapter params (the only base params with requires_grad=True)
    if args.use_lora:
        optim_params += [p for _, p in src_model.base.named_parameters() if p.requires_grad]
        optim_params += [p for _, p in tgt_model.base.named_parameters() if p.requires_grad]
    else:
        # Fallback: allow full-base finetune only if user explicitly disables freeze
        if not args.freeze_base:
            optim_params += [p for p in src_model.base.parameters() if p.requires_grad]
            optim_params += [p for p in tgt_model.base.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)
    scheduler = None

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        LOGGER.info(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            src_model, tgt_model, train_loader, optimizer, device, scheduler, args.max_grad_norm,
            loss_type=args.loss_type, temperature=args.temperature, alpha=args.alpha,
            src_max_len=args.src_max_len, tgt_max_len=args.tgt_max_len,
        )
        LOGGER.info(f"Train loss (avg): {train_loss:.6f}")

        if valid_loader is not None:
            val_loss = evaluate(
                src_model, tgt_model, valid_loader, device,
                loss_type=args.loss_type, temperature=args.temperature, alpha=args.alpha,
                src_max_len=args.src_max_len, tgt_max_len=args.tgt_max_len,
            )
            LOGGER.info(f"Validation loss (avg): {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                save_models(src_model, tgt_model, args.output_dir, save_base=args.save_base)
                LOGGER.info(f"Saved best models to {args.output_dir}")
        else:
            save_models(src_model, tgt_model, args.output_dir, save_base=args.save_base)

    LOGGER.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Bahnaric -> Vietnamese embedding models with MSE/InfoNCE")
    parser.add_argument("--train_csv", required=True, help="path to train CSV with columns Bahnaric, Vietnamese")
    parser.add_argument("--valid_csv", default=None, help="path to validation CSV (optional)")
    parser.add_argument("--src_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--tgt_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--freeze_base", action="store_true", help="freeze base encoders")
    parser.add_argument("--save_base", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="../results/models/translation-vn-bahna")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # NEW: loss control
    parser.add_argument("--loss_type", choices=["mse", "infonce", "hybrid"], default="infonce")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight: alpha*InfoNCE + (1-alpha)*MSE")

    # DataLoader options
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    # LoRA options
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters in the base encoders")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["query", "key", "value", "dense"], help="Module name substrings to apply LoRA to (depends on model)")


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    main(args)
