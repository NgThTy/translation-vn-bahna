#!/usr/bin/env python3
"""Train XLM-R LoRA + projection checkpoints on train_fit and evaluate on dev.

This script never reads the held-out test set. It trains a checkpoint using only
``--train_csv`` (normally ``data/train_fit.csv``), writes a provenance manifest,
and then calls ``src/previous_pipeline_baseline.py`` to evaluate one or both
retrieval rules on ``--dev_csv``. The resulting development ``metrics.json``
files participate in the same XLM-R LoRA family selection as the Kabsch and
pooling ablations.

The script also supports epoch-boundary resume checkpoints, which is useful on
clusters with short wall-time limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")
TRAINING_POLICY = "train_fit_only_no_dev_refit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    if lowercase:
        text = text.lower()
    return " ".join(text.split())


def read_parallel_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


class ParallelDataset(Dataset[Tuple[str, str]]):
    def __init__(
        self,
        csv_path: Path,
        lowercase: bool,
        sample_size: Optional[int],
        seed: int,
    ) -> None:
        df = read_parallel_csv(csv_path)
        if sample_size is not None and sample_size > 0 and sample_size < len(df):
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(df), size=sample_size, replace=False))
            df = df.iloc[indices].reset_index(drop=True)
        self.src = [
            normalize_text(value, lowercase=lowercase)
            for value in df["Bahnaric"].astype(str).tolist()
        ]
        self.tgt = [
            normalize_text(value, lowercase=lowercase)
            for value in df["Vietnamese"].astype(str).tolist()
        ]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.src[index], self.tgt[index]


def collate_batch(
    batch: Sequence[Tuple[str, str]],
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    src_max_len: int,
    tgt_max_len: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    source_texts, target_texts = zip(*batch)
    source_inputs = src_tokenizer(
        list(source_texts),
        padding=True,
        truncation=True,
        max_length=src_max_len,
        return_tensors="pt",
    )
    target_inputs = tgt_tokenizer(
        list(target_texts),
        padding=True,
        truncation=True,
        max_length=tgt_max_len,
        return_tensors="pt",
    )
    return source_inputs, target_inputs


def move_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_hidden_size(model: nn.Module) -> int:
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config, "dim", None)
    if hidden is None:
        raise ValueError("Could not infer hidden size from model config")
    return int(hidden)


def infer_lora_target_modules(base_model: nn.Module) -> List[str]:
    module_names = [name for name, _ in base_model.named_modules()]
    bert_style = ["query", "key", "value", "dense"]
    roberta_style = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    bert_matches = sum(
        1
        for name in module_names
        if any(part in name.split(".") for part in bert_style)
    )
    roberta_matches = sum(
        1
        for name in module_names
        if any(part in name.split(".") for part in roberta_style)
    )
    if bert_matches > 0:
        return bert_style
    if roberta_matches > 0:
        return roberta_style
    raise RuntimeError(
        "Could not infer LoRA target modules. Pass --lora_target_modules explicitly."
    )


def apply_lora(base_model: nn.Module, args: argparse.Namespace) -> nn.Module:
    target_modules = args.lora_target_modules or infer_lora_target_modules(base_model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base_model, config)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    print(f"LoRA target modules: {target_modules}")
    return model


def load_trainable_lora(base_model: nn.Module, adapter_dir: Path) -> nn.Module:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=True,
    )
    return model


def encode_batch(
    base_model: nn.Module,
    projection_head: nn.Module,
    inputs: Dict[str, torch.Tensor],
    pooling: str,
) -> torch.Tensor:
    outputs = base_model(**inputs, return_dict=True)
    hidden = outputs.last_hidden_state
    attention = inputs["attention_mask"].float()

    if pooling == "sentence_mean":
        mask = attention.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return projection_head(pooled)

    if pooling == "token_mean":
        batch_size, sequence_length, hidden_size = hidden.shape
        projected = projection_head(
            hidden.reshape(batch_size * sequence_length, hidden_size)
        ).reshape(batch_size, sequence_length, -1)
        mask = attention.unsqueeze(-1)
        return (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    raise ValueError(f"Unknown training pooling mode: {pooling}")


def symmetric_infonce(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    source = F.normalize(source_embeddings, p=2, dim=-1)
    target = F.normalize(target_embeddings, p=2, dim=-1)
    logits = (source @ target.T) / max(temperature, 1e-8)
    labels = torch.arange(source.size(0), device=source.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def save_model_checkpoint(
    source_model: nn.Module,
    target_model: nn.Module,
    source_projection: nn.Module,
    target_projection: nn.Module,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(source_projection.state_dict(), output_dir / "src_proj.pt")
    torch.save(target_projection.state_dict(), output_dir / "tgt_proj.pt")
    source_model.save_pretrained(output_dir / "src_adapters")
    target_model.save_pretrained(output_dir / "tgt_adapters")


def save_resume_state(
    output_dir: Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        output_dir / "training_state.pt",
    )


def write_training_manifest(
    checkpoint_dir: Path,
    args: argparse.Namespace,
    train_path: Path,
    train_pairs: int,
    completed_epochs: int,
) -> Path:
    manifest = {
        "schema_version": 1,
        "training_policy": TRAINING_POLICY,
        "train_csv": str(train_path),
        "train_csv_sha256": sha256_file(train_path),
        "num_train_pairs": int(train_pairs),
        "train_sample_size": (
            int(args.train_sample_size)
            if args.train_sample_size is not None and args.train_sample_size > 0
            else None
        ),
        "epochs": int(args.epochs),
        "completed_epochs": int(completed_epochs),
        "seed": int(args.seed),
        "src_model": args.src_model,
        "tgt_model": args.tgt_model,
        "pooling": args.pooling,
        "projection_dim": int(args.projection_dim),
        "projection_dropout": float(args.proj_dropout),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "temperature": float(args.temperature),
        "max_grad_norm": float(args.max_grad_norm),
        "lora_r": int(args.lora_r),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": args.lora_target_modules,
        "src_max_len": int(args.src_max_len),
        "tgt_max_len": int(args.tgt_max_len),
    }
    path = checkpoint_dir / "training_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_dev_evaluation(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    checkpoint_manifest: Path,
    retrieval: str,
) -> None:
    configuration_name = f"{args.configuration_prefix}_{retrieval}"
    output_dir = Path(args.dev_output_root) / configuration_name
    if (output_dir / "metrics.json").is_file() and not args.overwrite_dev_results:
        print(f"Skipping completed dev result: {configuration_name}")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        sys.executable,
        "src/previous_pipeline_baseline.py",
        "--input_csv",
        args.dev_csv,
        "--split_name",
        "dev",
        "--configuration_name",
        configuration_name,
        "--output_dir",
        str(output_dir),
        "--proj_dir",
        str(checkpoint_dir),
        "--checkpoint_manifest",
        str(checkpoint_manifest),
        "--src_model",
        args.src_model,
        "--tgt_model",
        args.tgt_model,
        "--pooling",
        args.pooling,
        "--batch_size",
        str(args.eval_batch_size),
        "--src_max_len",
        str(args.src_max_len),
        "--tgt_max_len",
        str(args.tgt_max_len),
        "--topk_eval",
        str(args.topk_eval),
        "--eval_ks",
        *[str(k) for k in args.eval_ks],
        "--csls_k",
        str(args.csls_k),
        "--use_lora",
        "--no_kabsch",
    ]
    if retrieval == "csls":
        command.append("--use_csls")
    if args.lowercase:
        command.append("--lowercase")
    if args.embedding_cache_dir:
        command.extend(["--embedding_cache_dir", args.embedding_cache_dir])
    if args.no_cuda:
        command.append("--no_cuda")
    if args.no_mps:
        command.append("--no_mps")

    print("Running development evaluation:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_path = Path(args.train_csv)
    dev_path = Path(args.dev_csv)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not dev_path.is_file():
        raise FileNotFoundError(f"Development CSV not found: {dev_path}")
    if sha256_file(train_path) == sha256_file(dev_path):
        raise ValueError("Training and development CSVs must be different files")

    if checkpoint_dir.exists() and args.overwrite_checkpoint and not args.resume_from:
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "mps"
        if hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not args.no_mps
        else "cpu"
    )
    print(f"Using device: {device}")

    source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    source_base_raw = AutoModel.from_pretrained(args.src_model)
    target_base_raw = AutoModel.from_pretrained(args.tgt_model)
    for parameter in source_base_raw.parameters():
        parameter.requires_grad = False
    for parameter in target_base_raw.parameters():
        parameter.requires_grad = False

    start_epoch = 1
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path is not None:
        source_model = load_trainable_lora(
            source_base_raw, resume_path / "src_adapters"
        ).to(device)
        target_model = load_trainable_lora(
            target_base_raw, resume_path / "tgt_adapters"
        ).to(device)
    else:
        source_model = apply_lora(source_base_raw, args).to(device)
        target_model = apply_lora(target_base_raw, args).to(device)

    source_projection = ProjectionHead(
        get_hidden_size(source_model),
        out_dim=args.projection_dim,
        dropout=args.proj_dropout,
    ).to(device)
    target_projection = ProjectionHead(
        get_hidden_size(target_model),
        out_dim=args.projection_dim,
        dropout=args.proj_dropout,
    ).to(device)

    if resume_path is not None:
        source_projection.load_state_dict(
            torch.load(resume_path / "src_proj.pt", map_location=device),
            strict=True,
        )
        target_projection.load_state_dict(
            torch.load(resume_path / "tgt_proj.pt", map_location=device),
            strict=True,
        )

    training_dataset = ParallelDataset(
        train_path,
        lowercase=args.lowercase,
        sample_size=args.train_sample_size,
        seed=args.seed,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_batch(
            batch,
            source_tokenizer,
            target_tokenizer,
            args.src_max_len,
            args.tgt_max_len,
        ),
    )

    trainable_parameters: List[nn.Parameter] = []
    trainable_parameters.extend(
        parameter for parameter in source_model.parameters() if parameter.requires_grad
    )
    trainable_parameters.extend(
        parameter for parameter in target_model.parameters() if parameter.requires_grad
    )
    trainable_parameters.extend(source_projection.parameters())
    trainable_parameters.extend(target_projection.parameters())

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(training_loader) / max(1, args.grad_accum_steps)
    )
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    if resume_path is not None:
        state_path = resume_path / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")

    final_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        source_model.train()
        target_model.train()
        source_projection.train()
        target_projection.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, (source_inputs, target_inputs) in enumerate(progress, start=1):
            source_inputs = move_to_device(source_inputs, device)
            target_inputs = move_to_device(target_inputs, device)
            source_embeddings = encode_batch(
                source_model,
                source_projection,
                source_inputs,
                args.pooling,
            )
            target_embeddings = encode_batch(
                target_model,
                target_projection,
                target_inputs,
                args.pooling,
            )
            loss = symmetric_infonce(
                source_embeddings,
                target_embeddings,
                args.temperature,
            )
            scaled_loss = loss / max(1, args.grad_accum_steps)
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())

            should_step = (
                step % args.grad_accum_steps == 0 or step == len(training_loader)
            )
            if should_step:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        epoch_checkpoint = checkpoint_dir / f"checkpoint_epoch_{epoch}"
        save_model_checkpoint(
            source_model,
            target_model,
            source_projection,
            target_projection,
            epoch_checkpoint,
        )
        save_resume_state(epoch_checkpoint, epoch, optimizer, scheduler)
        write_training_manifest(
            epoch_checkpoint,
            args,
            train_path,
            len(training_dataset),
            completed_epochs=epoch,
        )
        final_completed_epoch = epoch
        print(f"Saved resumable checkpoint: {epoch_checkpoint}")

    if final_completed_epoch < args.epochs:
        raise RuntimeError(
            f"Training stopped at epoch {final_completed_epoch}; expected {args.epochs}"
        )

    final_checkpoint = checkpoint_dir / "checkpoint_final"
    if final_checkpoint.exists():
        shutil.rmtree(final_checkpoint)
    save_model_checkpoint(
        source_model,
        target_model,
        source_projection,
        target_projection,
        final_checkpoint,
    )
    final_manifest = write_training_manifest(
        final_checkpoint,
        args,
        train_path,
        len(training_dataset),
        completed_epochs=final_completed_epoch,
    )
    print(f"Saved final checkpoint: {final_checkpoint}")

    retrievals = (
        ["cosine", "csls"]
        if args.eval_retrievals == "both"
        else [args.eval_retrievals]
    )
    for retrieval in retrievals:
        run_dev_evaluation(
            args,
            checkpoint_dir=final_checkpoint,
            checkpoint_manifest=final_manifest,
            retrieval=retrieval,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoRA + projection on train_fit and evaluate only on dev"
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dev_output_root", default="results/dev/xlmr_lora_projection")
    parser.add_argument("--configuration_prefix", required=True)
    parser.add_argument("--overwrite_checkpoint", action="store_true")
    parser.add_argument("--overwrite_dev_results", action="store_true")
    parser.add_argument("--resume_from", default=None)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")
    parser.add_argument(
        "--pooling",
        choices=["sentence_mean", "token_mean"],
        default="token_mean",
    )
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument(
        "--eval_retrievals",
        choices=["cosine", "csls", "both"],
        default="both",
    )
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_mps", action="store_true")
    main(parser.parse_args())
