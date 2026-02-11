"""
Fine-tune Bahnaric -> Vietnamese projection heads.
This script fine-tunes two embedding models (Vietnamese and Bahna) to produce aligned embeddings usable for cross-lingual retrieval and translation. It implements a practical version of the objective from Eq. (10): L(θ, γ) = E[ || f(X^γ) - Y^θ ||^2 ]
Practical choices in this script:
- Uses a lightweight multilingual Sentence-Transformer (default) for both sides.
- Adds small projection heads that map model outputs to a shared space.
- Trains projection heads (and optionally few lower layers) with MSE loss.
- Saves trained projection heads and optionally saves fine-tuned base models.
Input data:
- CSV files with columns: `Bahnaric` (Bahnaric text) and `Vietnamese` (Vietnamese text).
Example usage:
python train_embeddings.py \
--train_csv ../data/train.csv \
--src_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
--tgt_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
--projection_dim 256 \
--freeze_base False \
--output_dir ../results/models_demo \
--epochs 1 --batch_size 8 --lr 2e-4 \
--src_max_len 256 --tgt_max_len 256 \
--output_dir ../results/models/translation-vn-bahna
Requirements (from requirements.txt):
- torch transformers sentence-transformers pandas tqdm

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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Use a robust tqdm that counts batches
from tqdm import tqdm
import sys

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
        self.src_texts = df["Bahnaric"].astype(str).tolist()  # source = Bahnaric
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

    # Move to device
    for k in src_inputs:
        src_inputs[k] = src_inputs[k].to(device)
    for k in tgt_inputs:
        tgt_inputs[k] = tgt_inputs[k].to(device)
    return src_inputs, tgt_inputs


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class FineTuneModel(nn.Module):
    def __init__(self, base_model_name: str, projection_dim: int = 256, freeze_base: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)

        # Load model; many sentence-transformers are available under HF as AutoModel
        self.base = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            # Some models use pooler size or sentence-transformer wrappers
            hidden_size = getattr(self.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Cannot determine hidden size from config; check model name")

        self.proj = ProjectionHead(hidden_size, out_dim=projection_dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        # input_dict contains input_ids, attention_mask (optional token_type_ids)
        outputs = self.base(**input_dict, return_dict=True)

        # mean pooling over the sequence length
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        mask = input_dict.get("attention_mask", torch.ones(last_hidden.size()[:2], device=last_hidden.device))
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths  # (B, H)
        projected = self.proj(pooled)  # (B, proj_dim)
        return projected


def build_models(src_model_name, tgt_model_name, proj_dim, freeze_base, device):
    src = FineTuneModel(src_model_name, proj_dim, freeze_base).to(device)
    tgt = FineTuneModel(tgt_model_name, proj_dim, freeze_base).to(device)
    return src, tgt


# Use a robust tqdm that counts batches
def _make_pbar(iterable, desc):
    total = len(iterable)  # number of batches
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="batch",
        leave=False,
        ncols=100,
        disable=not sys.stdout.isatty(),
    )


def train_one_epoch(src_model, tgt_model, dataloader, optimizer, device, scheduler=None, max_grad_norm=None):
    src_model.train()
    tgt_model.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    pbar = _make_pbar(dataloader, "Training")
    for step, (src_inputs, tgt_inputs) in enumerate(pbar, 1):
        optimizer.zero_grad(set_to_none=True)

        src_emb = src_model(src_inputs)
        tgt_emb = tgt_model(tgt_inputs)
        loss = mse(src_emb, tgt_emb)

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                max_grad_norm,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss)
        if step % 50 == 0 or step == len(dataloader):
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(src_model, tgt_model, dataloader, device):
    src_model.eval()
    tgt_model.eval()
    total_loss = 0.0
    n_batches = 0
    mse = nn.MSELoss(reduction="mean")

    pbar = _make_pbar(dataloader, "Evaluating")
    for step, (src_inputs, tgt_inputs) in enumerate(pbar, 1):
        loss = mse(src_model(src_inputs), tgt_model(tgt_inputs))
        total_loss += float(loss)
        if step % 50 == 0 or step == len(dataloader):
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

    return total_loss / max(1, len(dataloader))


def save_models(src_model, tgt_model, out_dir: str, save_base: bool = False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Save projection heads
    torch.save(src_model.proj.state_dict(), Path(out_dir) / "src_proj.pt")
    torch.save(tgt_model.proj.state_dict(), Path(out_dir) / "tgt_proj.pt")
    # Save tokenizer names and model IDs for reproducibility
    with open(Path(out_dir) / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"saved_at: {Path(out_dir).absolute()}\n")
    # Optionally save base models if fine-tuned
    if save_base:
        try:
            src_model.base.save_pretrained(Path(out_dir) / "src_base")
        except Exception as e:
            LOGGER.warning(f"Could not save src base: {e}")
        try:
            tgt_model.base.save_pretrained(Path(out_dir) / "tgt_base")
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
        # Make the DataLoader Windows safe and configurable
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
            # Make the DataLoader Windows safe and configurable
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

    # Collect parameters to optimize: projection heads + (optionally) base model params
    optim_params = list(src_model.proj.parameters()) + list(tgt_model.proj.parameters())
    if not args.freeze_base:
        optim_params += [p for p in src_model.base.parameters() if p.requires_grad]
        optim_params += [p for p in tgt_model.base.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    # Optionally scheduler
    scheduler = None

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        LOGGER.info(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            src_model, tgt_model, train_loader, optimizer, device, scheduler, args.max_grad_norm
        )
        LOGGER.info(f"Train loss: {train_loss:.6f}")

        if valid_loader is not None:
            val_loss = evaluate(src_model, tgt_model, valid_loader, device)
            LOGGER.info(f"Validation loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_models(src_model, tgt_model, args.output_dir, save_base=args.save_base)
                LOGGER.info(f"Saved best models to {args.output_dir}")
        else:
            # save last epoch
            save_models(src_model, tgt_model, args.output_dir, save_base=args.save_base)
    LOGGER.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Bahnaric -> Vietnamese embedding models with MSE loss")
    parser.add_argument("--train_csv", required=True, help="path to train CSV with columns Bahnaric, Vietnamese")
    parser.add_argument("--valid_csv", default=None, help="path to validation CSV (optional)")
    parser.add_argument(
        "--src_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Hugging Face name for source model (Bahnaric)"
    )
    parser.add_argument(
        "--tgt_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Hugging Face name for target model (Vietnamese)"
    )
    parser.add_argument("--projection_dim", type=int, default=256, help="dimension of projection head output")
    parser.add_argument("--freeze_base", action="store_true", help="freeze pretrained base model weights (only train projection)")
    parser.add_argument("--save_base", action="store_true", help="save base models at the end if they were fine-tuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="../results/models/translation-vn-bahna")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda even if available")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max grad norm for clipping (optional)")

    # --- add to argparse ---
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows/CPU)")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for CUDA")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    main(args)
