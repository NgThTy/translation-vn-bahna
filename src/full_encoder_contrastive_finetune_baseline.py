# """
# Baseline 7: Full encoder contrastive fine-tuning with projection heads.

# Improved version.

# Main improvements:
# 1. Add projection heads, making this baseline closer to Baseline 6:
#    XLM-R -> pooling/projection -> 256-d retrieval space -> InfoNCE.

# 2. Support two pooling modes:
#    - sentence_mean:
#        encoder -> mean pool -> projection head
#    - token_mean:
#        encoder -> projection head per token -> mean pool

# 3. Train once, then evaluate the same checkpoint with both cosine and CSLS.

# 4. Support optional validation retrieval:
#    if --valid_csv is provided, after each epoch evaluate validation retrieval
#    and save the best checkpoint by validation MRR.

# 5. Recommended safer full-finetuning setup:
#    lr = 1e-5
#    batch_size = 4
#    grad_accum_steps = 16
#    effective batch size = 64
# """

# import argparse
# import json
# import math
# import random
# import re
# import shutil
# import unicodedata
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# # ============================================================
# # Reproducibility
# # ============================================================

# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # ============================================================
# # Text normalization
# # ============================================================

# def normalize_text(
#     text: str,
#     lowercase: bool = False,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
# ) -> str:
#     text = str(text)
#     text = unicodedata.normalize("NFC", text)

#     text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
#     text = text.replace("“", '"').replace("”", '"')
#     text = text.replace("–", "-").replace("—", "-")

#     if lowercase:
#         text = text.lower()

#     if strip_accents:
#         text = unicodedata.normalize("NFD", text)
#         text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
#         text = unicodedata.normalize("NFC", text)

#     if remove_punct:
#         text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# # ============================================================
# # Dataset
# # ============================================================

# class BahVnDataset(Dataset):
#     def __init__(
#         self,
#         csv_path: str,
#         lowercase: bool = False,
#         strip_accents: bool = False,
#         remove_punct: bool = False,
#     ):
#         df = pd.read_csv(csv_path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#         if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
#             raise ValueError("CSV must contain columns: Bahnaric,Vietnamese")

#         self.src_texts = [
#             normalize_text(
#                 x,
#                 lowercase=lowercase,
#                 strip_accents=strip_accents,
#                 remove_punct=remove_punct,
#             )
#             for x in df["Bahnaric"].astype(str).tolist()
#         ]

#         self.tgt_texts = [
#             normalize_text(
#                 x,
#                 lowercase=lowercase,
#                 strip_accents=strip_accents,
#                 remove_punct=remove_punct,
#             )
#             for x in df["Vietnamese"].astype(str).tolist()
#         ]

#     def __len__(self) -> int:
#         return len(self.src_texts)

#     def __getitem__(self, idx: int) -> Tuple[str, str]:
#         return self.src_texts[idx], self.tgt_texts[idx]


# def collate_batch(
#     batch,
#     src_tok,
#     tgt_tok,
#     src_max_len: int,
#     tgt_max_len: int,
# ):
#     src_texts, tgt_texts = zip(*batch)

#     src_inputs = src_tok(
#         list(src_texts),
#         padding=True,
#         truncation=True,
#         max_length=src_max_len,
#         return_tensors="pt",
#     )

#     tgt_inputs = tgt_tok(
#         list(tgt_texts),
#         padding=True,
#         truncation=True,
#         max_length=tgt_max_len,
#         return_tensors="pt",
#     )

#     return src_inputs, tgt_inputs


# def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
#     return {k: v.to(device) for k, v in batch.items()}


# # ============================================================
# # Model components
# # ============================================================

# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)


# def get_hidden_size(model) -> int:
#     hidden = getattr(model.config, "hidden_size", None)
#     if hidden is None:
#         hidden = getattr(model.config, "dim", None)
#     if hidden is None:
#         raise ValueError("Could not infer hidden size from model config.")
#     return int(hidden)


# def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     mask = attention_mask.unsqueeze(-1).float()
#     summed = (last_hidden * mask).sum(dim=1)
#     denom = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / denom


# def cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
#     return last_hidden[:, 0, :]


# def encode_batch(
#     model,
#     proj_head,
#     inputs: Dict[str, torch.Tensor],
#     pooling: str,
# ) -> torch.Tensor:
#     outputs = model(**inputs, return_dict=True)
#     last = outputs.last_hidden_state
#     attn = inputs["attention_mask"].float()

#     if pooling == "sentence_mean":
#         pooled = mean_pool(last, inputs["attention_mask"])
#         emb = proj_head(pooled)
#         return emb

#     if pooling == "cls":
#         pooled = cls_pool(last)
#         emb = proj_head(pooled)
#         return emb

#     if pooling == "token_mean":
#         bsz, seq_len, hidden = last.shape
#         flat = last.reshape(bsz * seq_len, hidden)
#         projected = proj_head(flat).reshape(bsz, seq_len, -1)

#         mask = attn.unsqueeze(-1)
#         emb = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#         return emb

#     raise ValueError(f"Unknown pooling mode: {pooling}")


# # ============================================================
# # Loss
# # ============================================================

# def symmetric_infonce(src_emb: torch.Tensor, tgt_emb: torch.Tensor, temperature: float) -> torch.Tensor:
#     src = F.normalize(src_emb, p=2, dim=-1)
#     tgt = F.normalize(tgt_emb, p=2, dim=-1)

#     logits = src @ tgt.T
#     logits = logits / max(temperature, 1e-8)

#     labels = torch.arange(src.size(0), device=src.device)

#     loss_src = F.cross_entropy(logits, labels)
#     loss_tgt = F.cross_entropy(logits.T, labels)

#     return 0.5 * (loss_src + loss_tgt)


# # ============================================================
# # Retrieval helpers
# # ============================================================

# def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return x / norms


# def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)

#     sims = q @ z.T
#     k = int(max(1, min(topk, index.shape[0])))
#     topk_idx = np.argsort(-sims, axis=1)[:, :k]

#     return topk_idx, sims


# def csls_topk(query: np.ndarray, index: np.ndarray, topk: int, csls_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)
#     sims = q @ z.T

#     if q.shape[0] <= 1 or z.shape[0] <= 1:
#         return cosine_topk(query, index, topk)

#     k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))

#     rq = np.partition(sims, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
#     rz = np.partition(sims, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)

#     csls = 2.0 * sims - rq[:, None] - rz[None, :]

#     k = int(max(1, min(topk, index.shape[0])))
#     topk_idx = np.argsort(-csls, axis=1)[:, :k]

#     return topk_idx, csls


# def ranking_metrics(
#     topk_idx: np.ndarray,
#     gold_idx: np.ndarray,
#     eval_ks: List[int],
# ) -> Tuple[Dict[str, float], np.ndarray]:
#     n_items = topk_idx.shape[0]
#     ranks = np.full(n_items, np.inf, dtype=np.float64)

#     for i in range(n_items):
#         hits = np.where(topk_idx[i] == gold_idx[i])[0]
#         if len(hits) > 0:
#             ranks[i] = float(hits[0] + 1)

#     metrics: Dict[str, float] = {}
#     metrics["MRR"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
#     metrics["Top1_acc"] = float(np.mean(ranks == 1.0))

#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         hit = float(np.mean(ranks <= k_eff))
#         metrics[f"Hit@{k_eff}"] = hit
#         metrics[f"Recall@{k_eff}"] = hit
#         metrics[f"Precision@{k_eff}"] = hit / float(k_eff)

#     return metrics, ranks


# def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
#     df_out = df_out.copy()

#     df_out["Bahnaric_len_chars"] = df_out["Bahnaric"].astype(str).str.len()
#     df_out["Vietnamese_len_chars"] = df_out["Gold_VN"].astype(str).str.len()
#     df_out["Vietnamese_len_words"] = df_out["Gold_VN"].astype(str).str.split().map(len)

#     def vn_len_bin(n: int) -> str:
#         if n <= 5:
#             return "1-5"
#         if n <= 15:
#             return "6-15"
#         if n <= 30:
#             return "16-30"
#         return ">30"

#     df_out["VN_len_bin"] = df_out["Vietnamese_len_words"].map(vn_len_bin)
#     return df_out


# def save_bucket_metrics(df_out: pd.DataFrame, output_dir: Path, eval_ks: List[int]) -> None:
#     metric_cols = ["P@1", "MRR"] + [f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns]
#     available_cols = [col for col in metric_cols if col in df_out.columns]

#     grouped = df_out.groupby("VN_len_bin", dropna=False)
#     bucket_df = grouped[available_cols].mean().reset_index()
#     bucket_df["count"] = grouped.size().values

#     for col in available_cols:
#         bucket_df[col] = bucket_df[col].astype(float).round(4)

#     bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


# # ============================================================
# # Encoding and evaluation
# # ============================================================

# @torch.no_grad()
# def encode_texts(
#     texts: List[str],
#     tokenizer,
#     model,
#     proj_head,
#     device: torch.device,
#     max_len: int,
#     batch_size: int,
#     pooling: str,
#     use_amp: bool = False,
#     amp_dtype: torch.dtype = torch.float16,
# ) -> np.ndarray:
#     model.eval()
#     proj_head.eval()
#     all_embs = []

#     for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
#         batch = texts[start : start + batch_size]

#         inputs = tokenizer(
#             batch,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         inputs = move_to_device(inputs, device)

#         with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
#             emb = encode_batch(model, proj_head, inputs, pooling=pooling)

#         all_embs.append(emb.detach().float().cpu().numpy())

#     return np.vstack(all_embs).astype(np.float32)


# def evaluate_retrieval(
#     csv_path: str,
#     src_tok,
#     tgt_tok,
#     src_model,
#     tgt_model,
#     src_proj,
#     tgt_proj,
#     device: torch.device,
#     output_dir: Path,
#     args,
#     retrieval: str,
#     split_name: str,
#     save_predictions: bool = True,
# ) -> Dict[str, float]:
#     df = pd.read_csv(csv_path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#     raw_bah = df["Bahnaric"].astype(str).tolist()
#     raw_vn = df["Vietnamese"].astype(str).tolist()

#     bah = [
#         normalize_text(x, lowercase=args.lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
#         for x in raw_bah
#     ]
#     vn = [
#         normalize_text(x, lowercase=args.lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
#         for x in raw_vn
#     ]

#     eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
#     use_amp = bool(args.fp16 or args.bf16) and device.type == "cuda"
#     amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

#     print(f"[Eval:{split_name}:{retrieval}] Encoding Bahnaric queries...")
#     bah_emb = encode_texts(
#         bah,
#         src_tok,
#         src_model,
#         src_proj,
#         device,
#         max_len=args.src_max_len,
#         batch_size=eval_batch_size,
#         pooling=args.pooling,
#         use_amp=use_amp,
#         amp_dtype=amp_dtype,
#     )

#     print(f"[Eval:{split_name}:{retrieval}] Encoding Vietnamese candidates...")
#     vn_emb = encode_texts(
#         vn,
#         tgt_tok,
#         tgt_model,
#         tgt_proj,
#         device,
#         max_len=args.tgt_max_len,
#         batch_size=eval_batch_size,
#         pooling=args.pooling,
#         use_amp=use_amp,
#         amp_dtype=amp_dtype,
#     )

#     if retrieval == "csls":
#         topk_idx, scores = csls_topk(
#             query=bah_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#     elif retrieval == "cosine":
#         topk_idx, scores = cosine_topk(
#             query=bah_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#         )
#     else:
#         raise ValueError(f"Unknown retrieval mode: {retrieval}")

#     gold_idx = np.arange(len(df), dtype=np.int64)
#     eval_ks = [int(k) for k in args.eval_ks]

#     metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

#     rounded_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
#     rounded_metrics.update(
#         {
#             "method": "full_encoder_contrastive_finetuning_projection",
#             "pipeline": "full_encoder_projection_infonce_retrieval",
#             "split": split_name,
#             "csv_path": csv_path,
#             "src_model": args.src_model,
#             "tgt_model": args.tgt_model,
#             "pooling": args.pooling,
#             "projection_dim": int(args.projection_dim),
#             "retrieval": retrieval,
#             "num_queries": int(len(df)),
#             "candidate_pool_size": int(len(df)),
#             "embedding_dim": int(bah_emb.shape[1]),
#             "epochs": int(args.epochs),
#             "batch_size": int(args.batch_size),
#             "grad_accum_steps": int(args.grad_accum_steps),
#             "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
#             "lr": float(args.lr),
#             "weight_decay": float(args.weight_decay),
#             "temperature": float(args.temperature),
#             "warmup_ratio": float(args.warmup_ratio),
#             "src_max_len": int(args.src_max_len),
#             "tgt_max_len": int(args.tgt_max_len),
#             "topk_eval": int(args.topk_eval),
#             "eval_ks": eval_ks,
#             "use_csls": bool(retrieval == "csls"),
#             "csls_k": int(args.csls_k),
#             "fp16": bool(args.fp16),
#             "bf16": bool(args.bf16),
#             "strip_accents": bool(args.strip_accents),
#             "lowercase": bool(args.lowercase),
#             "remove_punct": bool(args.remove_punct),
#             "seed": int(args.seed),
#         }
#     )

#     eval_dir = output_dir / f"{split_name}_{retrieval}"
#     eval_dir.mkdir(parents=True, exist_ok=True)

#     with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
#         json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

#     if save_predictions:
#         pred_top1_idx = topk_idx[:, 0]
#         preds = [raw_vn[i] for i in pred_top1_idx]
#         topk_preds = ["|".join([raw_vn[j] for j in row]) for row in topk_idx]

#         is_top1 = (pred_top1_idx == gold_idx).astype(np.float32)
#         rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)
#         gold_rank = [None if not np.isfinite(r) else int(r) for r in ranks.tolist()]

#         out = {
#             "Bahnaric": raw_bah,
#             "Bahnaric_normalized": bah,
#             "Predicted_VN": preds,
#             "Gold_VN": raw_vn,
#             "Gold_VN_normalized": vn,
#             "Gold_rank": gold_rank,
#             "TopK_Preds": topk_preds,
#             "P@1": is_top1.tolist(),
#             "MRR": rr.tolist(),
#             "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
#             "Gold_score": [float(scores[i, i]) for i in range(len(df))],
#         }

#         for k in eval_ks:
#             k_eff = int(max(1, min(k, topk_idx.shape[1])))
#             out[f"Hit@{k_eff}"] = (ranks <= float(k_eff)).astype(np.float32).tolist()

#         df_out = pd.DataFrame(out)
#         df_out = make_bucket_columns(df_out)
#         df_out.to_csv(eval_dir / "sentence_predictions.csv", index=False)

#         save_bucket_metrics(df_out, eval_dir, eval_ks)

#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     return rounded_metrics


# # ============================================================
# # Checkpointing
# # ============================================================

# def save_checkpoint(
#     src_model,
#     tgt_model,
#     src_proj,
#     tgt_proj,
#     src_tok,
#     tgt_tok,
#     checkpoint_dir: Path,
# ) -> None:
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)

#     src_model.save_pretrained(checkpoint_dir / "src_encoder")
#     tgt_model.save_pretrained(checkpoint_dir / "tgt_encoder")
#     src_tok.save_pretrained(checkpoint_dir / "src_encoder")
#     tgt_tok.save_pretrained(checkpoint_dir / "tgt_encoder")

#     torch.save(src_proj.state_dict(), checkpoint_dir / "src_proj.pt")
#     torch.save(tgt_proj.state_dict(), checkpoint_dir / "tgt_proj.pt")


# def copy_best_to_root(best_dir: Path, output_dir: Path) -> None:
#     final_dir = output_dir / "best_checkpoint"
#     if final_dir.exists():
#         shutil.rmtree(final_dir)
#     shutil.copytree(best_dir, final_dir)


# # ============================================================
# # Main
# # ============================================================

# def main(args: argparse.Namespace) -> None:
#     set_seed(args.seed)

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

#     print(f"Using device: {device}")
#     print("Baseline: full_encoder_contrastive_finetuning_projection")
#     print(f"Source model: {args.src_model}")
#     print(f"Target model: {args.tgt_model}")
#     print(f"Pooling: {args.pooling}")
#     print(f"Projection dim: {args.projection_dim}")
#     print(f"Loss: symmetric InfoNCE")
#     print(f"Full encoder fine-tuning: true")

#     if device.type == "cpu":
#         print("WARNING: Running full encoder fine-tuning on CPU will be very slow. Use a GPU compute node.")

#     src_tok = AutoTokenizer.from_pretrained(args.src_model)
#     tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)

#     src_model = AutoModel.from_pretrained(args.src_model).to(device)
#     tgt_model = AutoModel.from_pretrained(args.tgt_model).to(device)

#     hidden_size_src = get_hidden_size(src_model)
#     hidden_size_tgt = get_hidden_size(tgt_model)

#     src_proj = ProjectionHead(hidden_size_src, out_dim=args.projection_dim, dropout=args.proj_dropout).to(device)
#     tgt_proj = ProjectionHead(hidden_size_tgt, out_dim=args.projection_dim, dropout=args.proj_dropout).to(device)

#     train_ds = BahVnDataset(
#         args.train_csv,
#         lowercase=args.lowercase,
#         strip_accents=args.strip_accents,
#         remove_punct=args.remove_punct,
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=False,
#         num_workers=args.num_workers,
#         pin_memory=(device.type == "cuda"),
#         collate_fn=lambda b: collate_batch(
#             b,
#             src_tok,
#             tgt_tok,
#             args.src_max_len,
#             args.tgt_max_len,
#         ),
#     )

#     trainable_params = (
#         list(src_model.parameters())
#         + list(tgt_model.parameters())
#         + list(src_proj.parameters())
#         + list(tgt_proj.parameters())
#     )

#     optimizer = torch.optim.AdamW(
#         trainable_params,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )

#     updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
#     total_steps = max(1, updates_per_epoch * args.epochs)
#     warmup_steps = int(args.warmup_ratio * total_steps)

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps,
#     )

#     use_amp = bool(args.fp16 or args.bf16) and device.type == "cuda"
#     amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
#     scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

#     best_valid_mrr = -1.0
#     best_epoch = -1
#     patience_counter = 0
#     best_dir = None

#     with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
#         json.dump(vars(args), f, ensure_ascii=False, indent=2)

#     for epoch in range(1, args.epochs + 1):
#         src_model.train()
#         tgt_model.train()
#         src_proj.train()
#         tgt_proj.train()

#         running_loss = 0.0
#         optimizer.zero_grad(set_to_none=True)

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

#         for step, (src_inputs, tgt_inputs) in enumerate(pbar, start=1):
#             src_inputs = move_to_device(src_inputs, device)
#             tgt_inputs = move_to_device(tgt_inputs, device)

#             with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
#                 src_emb = encode_batch(src_model, src_proj, src_inputs, pooling=args.pooling)
#                 tgt_emb = encode_batch(tgt_model, tgt_proj, tgt_inputs, pooling=args.pooling)
#                 loss = symmetric_infonce(src_emb, tgt_emb, temperature=args.temperature)
#                 loss = loss / max(1, args.grad_accum_steps)

#             if scaler.is_enabled():
#                 scaler.scale(loss).backward()
#             else:
#                 loss.backward()

#             running_loss += float(loss.detach().cpu()) * max(1, args.grad_accum_steps)

#             should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))

#             if should_step:
#                 if args.max_grad_norm > 0:
#                     if scaler.is_enabled():
#                         scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

#                 if scaler.is_enabled():
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     optimizer.step()

#                 scheduler.step()
#                 optimizer.zero_grad(set_to_none=True)

#             avg_loss = running_loss / max(1, step)
#             pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

#         epoch_loss = running_loss / max(1, len(train_loader))
#         print(f"Epoch {epoch} train_loss={epoch_loss:.6f}")

#         epoch_ckpt = output_dir / f"checkpoint_epoch_{epoch}"
#         save_checkpoint(src_model, tgt_model, src_proj, tgt_proj, src_tok, tgt_tok, epoch_ckpt)

#         if args.valid_csv:
#             print(f"Evaluating validation retrieval after epoch {epoch}...")
#             valid_metrics = evaluate_retrieval(
#                 csv_path=args.valid_csv,
#                 src_tok=src_tok,
#                 tgt_tok=tgt_tok,
#                 src_model=src_model,
#                 tgt_model=tgt_model,
#                 src_proj=src_proj,
#                 tgt_proj=tgt_proj,
#                 device=device,
#                 output_dir=output_dir / f"epoch_{epoch}_validation_eval",
#                 args=args,
#                 retrieval=args.early_stop_retrieval,
#                 split_name="valid",
#                 save_predictions=False,
#             )

#             valid_mrr = float(valid_metrics.get("MRR", 0.0))

#             if valid_mrr > best_valid_mrr:
#                 best_valid_mrr = valid_mrr
#                 best_epoch = epoch
#                 patience_counter = 0
#                 best_dir = epoch_ckpt
#                 copy_best_to_root(best_dir, output_dir)
#                 print(f"New best checkpoint: epoch={epoch}, valid_MRR={valid_mrr:.4f}")
#             else:
#                 patience_counter += 1
#                 print(f"No validation improvement. patience={patience_counter}/{args.early_stop_patience}")

#             if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
#                 print("Early stopping triggered.")
#                 break
#         else:
#             best_dir = epoch_ckpt
#             copy_best_to_root(best_dir, output_dir)

#     print(f"Training complete. Best epoch: {best_epoch if best_epoch > 0 else 'final'}")

#     # If validation was used, reload the best checkpoint for final test evaluation.
#     # For simplicity, continue with current in-memory model if no validation.
#     if args.valid_csv and (output_dir / "best_checkpoint").exists():
#         print("Loading best checkpoint for final test evaluation...")
#         src_model = AutoModel.from_pretrained(output_dir / "best_checkpoint" / "src_encoder").to(device)
#         tgt_model = AutoModel.from_pretrained(output_dir / "best_checkpoint" / "tgt_encoder").to(device)

#         src_proj.load_state_dict(torch.load(output_dir / "best_checkpoint" / "src_proj.pt", map_location=device))
#         tgt_proj.load_state_dict(torch.load(output_dir / "best_checkpoint" / "tgt_proj.pt", map_location=device))

#     print("Evaluating final checkpoint on test set...")

#     if args.eval_retrievals in {"cosine", "both"}:
#         evaluate_retrieval(
#             csv_path=args.test_csv,
#             src_tok=src_tok,
#             tgt_tok=tgt_tok,
#             src_model=src_model,
#             tgt_model=tgt_model,
#             src_proj=src_proj,
#             tgt_proj=tgt_proj,
#             device=device,
#             output_dir=output_dir,
#             args=args,
#             retrieval="cosine",
#             split_name="test",
#             save_predictions=True,
#         )

#     if args.eval_retrievals in {"csls", "both"}:
#         evaluate_retrieval(
#             csv_path=args.test_csv,
#             src_tok=src_tok,
#             tgt_tok=tgt_tok,
#             src_model=src_model,
#             tgt_model=tgt_model,
#             src_proj=src_proj,
#             tgt_proj=tgt_proj,
#             device=device,
#             output_dir=output_dir,
#             args=args,
#             retrieval="csls",
#             split_name="test",
#             save_predictions=True,
#         )

#     print(f"Saved output to {output_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Baseline 7: full encoder contrastive fine-tuning with projection heads"
#     )

#     parser.add_argument("--train_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for training")
#     parser.add_argument("--valid_csv", default=None, help="Optional validation CSV with columns Bahnaric,Vietnamese")
#     parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for retrieval evaluation")
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--src_model", default="xlm-roberta-base")
#     parser.add_argument("--tgt_model", default="xlm-roberta-base")

#     parser.add_argument(
#         "--pooling",
#         choices=["sentence_mean", "token_mean", "cls"],
#         default="sentence_mean",
#         help=(
#             "sentence_mean: mean pool encoder tokens then project; "
#             "token_mean: project each token then mean pool; "
#             "cls: use first token then project."
#         ),
#     )

#     parser.add_argument("--projection_dim", type=int, default=256)
#     parser.add_argument("--proj_dropout", type=float, default=0.1)

#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--eval_batch_size", type=int, default=None)
#     parser.add_argument("--grad_accum_steps", type=int, default=16)

#     parser.add_argument("--lr", type=float, default=1e-5)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--warmup_ratio", type=float, default=0.06)
#     parser.add_argument("--temperature", type=float, default=0.07)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)

#     parser.add_argument("--src_max_len", type=int, default=256)
#     parser.add_argument("--tgt_max_len", type=int, default=256)

#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument(
#         "--eval_retrievals",
#         choices=["cosine", "csls", "both"],
#         default="both",
#         help="Evaluate the same trained checkpoint with cosine, CSLS, or both.",
#     )
#     parser.add_argument("--csls_k", type=int, default=10)

#     parser.add_argument(
#         "--early_stop_retrieval",
#         choices=["cosine", "csls"],
#         default="csls",
#         help="Retrieval method used for validation early stopping.",
#     )
#     parser.add_argument("--early_stop_patience", type=int, default=0, help="0 disables early stopping.")

#     parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA.")
#     parser.add_argument("--bf16", action="store_true", help="Use bf16 autocast on CUDA. Prefer this on A100/H100 if supported.")

#     parser.add_argument("--lowercase", action="store_true")
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")

#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--no_cuda", action="store_true")

#     args = parser.parse_args()

#     if args.fp16 and args.bf16:
#         raise ValueError("Use only one of --fp16 or --bf16, not both.")

#     main(args)

#################################
#!/usr/bin/env python3
"""Reviewer-compliant full-encoder contrastive retrieval baseline.

Training mode:
- trains only on train_fit.csv;
- saves resumable checkpoints;
- evaluates declared checkpoints (for example, epochs 3, 5, and 10) only on dev.csv;
- writes one dev metrics.json per checkpoint/retrieval configuration.

Evaluation mode:
- loads one already-trained checkpoint;
- requires the configuration to match the shared development-selection manifest;
- evaluates the authorized configuration once on the held-out test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

FAMILY = "full_encoder"
EVALUATOR_SCRIPT = "src/full_encoder_contrastive_finetune_baseline.py"
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
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def normalize_text(
    text: str,
    lowercase: bool = False,
    strip_accents: bool = False,
    remove_punct: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    if lowercase:
        text = text.lower()
    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


class BahVnDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        lowercase: bool,
        strip_accents: bool,
        remove_punct: bool,
    ) -> None:
        df = pd.read_csv(csv_path)
        missing = [column for column in ("Bahnaric", "Vietnamese") if column not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        df = df.dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
        self.src_texts = [
            normalize_text(x, lowercase, strip_accents, remove_punct)
            for x in df["Bahnaric"].astype(str)
        ]
        self.tgt_texts = [
            normalize_text(x, lowercase, strip_accents, remove_punct)
            for x in df["Vietnamese"].astype(str)
        ]

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.src_texts[index], self.tgt_texts[index]


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: Iterable[int]) -> None:
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def collate_batch(batch, src_tok, tgt_tok, src_max_len: int, tgt_max_len: int):
    src_texts, tgt_texts = zip(*batch)
    src_inputs = src_tok(
        list(src_texts), padding=True, truncation=True,
        max_length=src_max_len, return_tensors="pt",
    )
    tgt_inputs = tgt_tok(
        list(tgt_texts), padding=True, truncation=True,
        max_length=tgt_max_len, return_tensors="pt",
    )
    return src_inputs, tgt_inputs


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class IdentityProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_hidden_size(model) -> int:
    value = getattr(model.config, "hidden_size", None)
    if value is None:
        value = getattr(model.config, "dim", None)
    if value is None:
        raise ValueError("Could not infer encoder hidden size")
    return int(value)


def build_projection(
    projection_type: str,
    hidden_size: int,
    projection_dim: int,
    dropout: float,
) -> Tuple[nn.Module, int]:
    if projection_type == "mlp":
        return MLPProjection(hidden_size, projection_dim, dropout), projection_dim
    if projection_type == "linear":
        return LinearProjection(hidden_size, projection_dim), projection_dim
    if projection_type == "identity":
        return IdentityProjection(), hidden_size
    raise ValueError(f"Unknown projection_type: {projection_type}")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_batch(model, projection, inputs: Dict[str, torch.Tensor], pooling: str) -> torch.Tensor:
    outputs = model(**inputs, return_dict=True)
    last = outputs.last_hidden_state
    attention = inputs["attention_mask"].float()

    if pooling == "sentence_mean":
        return projection(mean_pool(last, inputs["attention_mask"]))
    if pooling == "cls":
        return projection(last[:, 0, :])
    if pooling == "token_mean":
        batch_size, seq_len, hidden_size = last.shape
        projected = projection(last.reshape(batch_size * seq_len, hidden_size))
        projected = projected.reshape(batch_size, seq_len, -1)
        mask = attention.unsqueeze(-1)
        return (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"Unknown pooling: {pooling}")


def contrastive_loss(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    temperature: float,
    loss_type: str,
) -> torch.Tensor:
    src = F.normalize(src_emb, p=2, dim=-1)
    tgt = F.normalize(tgt_emb, p=2, dim=-1)
    logits = (src @ tgt.T) / max(temperature, 1e-8)
    labels = torch.arange(src.size(0), device=src.device)
    forward = F.cross_entropy(logits, labels)
    if loss_type == "source_to_target_infonce":
        return forward
    if loss_type == "symmetric_infonce":
        backward = F.cross_entropy(logits.T, labels)
        return 0.5 * (forward + backward)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = l2_normalize_rows(query) @ l2_normalize_rows(index).T
    k = max(1, min(int(topk), index.shape[0]))
    return np.argsort(-scores, axis=1)[:, :k], scores


def csls_topk(
    query: np.ndarray,
    index: np.ndarray,
    topk: int,
    csls_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    similarities = q @ z.T
    if q.shape[0] <= 1 or z.shape[0] <= 1:
        return cosine_topk(query, index, topk)
    k_csls = max(1, min(int(csls_k), q.shape[0] - 1, z.shape[0] - 1))
    rq = np.partition(similarities, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
    rz = np.partition(similarities, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)
    scores = 2.0 * similarities - rq[:, None] - rz[None, :]
    k = max(1, min(int(topk), index.shape[0]))
    return np.argsort(-scores, axis=1)[:, :k], scores


def ranking_metrics(
    topk_idx: np.ndarray,
    gold_idx: np.ndarray,
    eval_ks: List[int],
) -> Tuple[Dict[str, float], np.ndarray]:
    ranks = np.full(topk_idx.shape[0], np.inf, dtype=np.float64)
    for row in range(topk_idx.shape[0]):
        hits = np.where(topk_idx[row] == gold_idx[row])[0]
        if len(hits):
            ranks[row] = float(hits[0] + 1)
    metrics: Dict[str, float] = {
        "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "Top1_acc": float(np.mean(ranks == 1.0)),
    }
    for k in eval_ks:
        k_eff = max(1, min(int(k), topk_idx.shape[1]))
        hit = float(np.mean(ranks <= k_eff))
        metrics[f"Hit@{k_eff}"] = hit
        metrics[f"Recall@{k_eff}"] = hit
        metrics[f"Precision@{k_eff}"] = hit / float(k_eff)
    return metrics, ranks


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    projection,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pooling: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    projection.eval()
    outputs: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        inputs = move_to_device(inputs, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp and device.type == "cuda",
        ):
            emb = encode_batch(model, projection, inputs, pooling)
        outputs.append(emb.detach().float().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def read_eval_texts(args: argparse.Namespace, csv_path: str):
    df = pd.read_csv(csv_path)
    missing = [column for column in ("Bahnaric", "Vietnamese") if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
    raw_src = df["Bahnaric"].astype(str).tolist()
    raw_tgt = df["Vietnamese"].astype(str).tolist()
    src = [
        normalize_text(x, args.lowercase, args.strip_accents, args.remove_punct)
        for x in raw_src
    ]
    tgt = [
        normalize_text(x, args.lowercase, args.strip_accents, args.remove_punct)
        for x in raw_tgt
    ]
    return df, raw_src, raw_tgt, src, tgt


def encode_eval_pool(
    args: argparse.Namespace,
    csv_path: str,
    src_tok,
    tgt_tok,
    src_model,
    tgt_model,
    src_projection,
    tgt_projection,
    device: torch.device,
):
    df, raw_src, raw_tgt, src_texts, tgt_texts = read_eval_texts(args, csv_path)
    eval_batch_size = args.eval_batch_size or args.batch_size
    use_amp = bool(args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    src_emb = encode_texts(
        src_texts, src_tok, src_model, src_projection, device,
        args.src_max_len, eval_batch_size, args.pooling, use_amp, amp_dtype,
    )
    tgt_emb = encode_texts(
        tgt_texts, tgt_tok, tgt_model, tgt_projection, device,
        args.tgt_max_len, eval_batch_size, args.pooling, use_amp, amp_dtype,
    )
    return df, raw_src, raw_tgt, src_texts, tgt_texts, src_emb, tgt_emb


def evaluate_embeddings(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    retrieval: str,
    topk_eval: int,
    csls_k: int,
    eval_ks: List[int],
):
    if retrieval == "cosine":
        topk_idx, scores = cosine_topk(src_emb, tgt_emb, topk_eval)
    elif retrieval == "csls":
        topk_idx, scores = csls_topk(src_emb, tgt_emb, topk_eval, csls_k)
    else:
        raise ValueError(f"Unknown retrieval: {retrieval}")
    gold_idx = np.arange(src_emb.shape[0], dtype=np.int64)
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)
    return metrics, ranks, topk_idx, scores


def checkpoint_metadata(args: argparse.Namespace, checkpoint_epoch: int) -> Dict[str, Any]:
    train_path = Path(args.train_csv)
    return {
        "schema_version": 1,
        "family": FAMILY,
        "training_policy": TRAINING_POLICY,
        "train_csv": str(train_path),
        "train_csv_sha256": sha256_file(train_path),
        "src_model": args.src_model,
        "tgt_model": args.tgt_model,
        "pooling": args.pooling,
        "projection_type": args.projection_type,
        "projection_dim": int(args.projection_dim),
        "proj_dropout": float(args.proj_dropout),
        "loss_type": args.loss_type,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "warmup_ratio": float(args.warmup_ratio),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
        "src_max_len": int(args.src_max_len),
        "tgt_max_len": int(args.tgt_max_len),
        "lowercase": bool(args.lowercase),
        "strip_accents": bool(args.strip_accents),
        "remove_punct": bool(args.remove_punct),
        "seed": int(args.seed),
        "checkpoint_epoch": int(checkpoint_epoch),
    }


def save_checkpoint_directory(
    checkpoint_dir: Path,
    src_model,
    tgt_model,
    src_projection,
    tgt_projection,
    src_tok,
    tgt_tok,
    metadata: Dict[str, Any],
    training_state: Optional[Dict[str, Any]],
) -> None:
    temp_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    src_model.save_pretrained(temp_dir / "src_encoder")
    tgt_model.save_pretrained(temp_dir / "tgt_encoder")
    src_tok.save_pretrained(temp_dir / "src_encoder")
    tgt_tok.save_pretrained(temp_dir / "tgt_encoder")
    torch.save(src_projection.state_dict(), temp_dir / "src_projection.pt")
    torch.save(tgt_projection.state_dict(), temp_dir / "tgt_projection.pt")
    (temp_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if training_state is not None:
        torch.save(training_state, temp_dir / "training_state.pt")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    temp_dir.rename(checkpoint_dir)


def load_checkpoint_components(checkpoint_dir: Path, device: torch.device):
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    src_tok = AutoTokenizer.from_pretrained(checkpoint_dir / "src_encoder")
    tgt_tok = AutoTokenizer.from_pretrained(checkpoint_dir / "tgt_encoder")
    src_model = AutoModel.from_pretrained(checkpoint_dir / "src_encoder").to(device)
    tgt_model = AutoModel.from_pretrained(checkpoint_dir / "tgt_encoder").to(device)
    src_projection, _ = build_projection(
        metadata["projection_type"], get_hidden_size(src_model),
        int(metadata["projection_dim"]), float(metadata["proj_dropout"]),
    )
    tgt_projection, _ = build_projection(
        metadata["projection_type"], get_hidden_size(tgt_model),
        int(metadata["projection_dim"]), float(metadata["proj_dropout"]),
    )
    src_projection.load_state_dict(
        torch.load(checkpoint_dir / "src_projection.pt", map_location=device), strict=True
    )
    tgt_projection.load_state_dict(
        torch.load(checkpoint_dir / "tgt_projection.pt", map_location=device), strict=True
    )
    return (
        src_tok, tgt_tok, src_model, tgt_model,
        src_projection.to(device), tgt_projection.to(device), metadata,
    )


def build_configuration(
    metadata: Dict[str, Any],
    checkpoint_dir: Path,
    retrieval: str,
    csls_k: int,
) -> Dict[str, Any]:
    fields = [
        "training_policy", "train_csv", "train_csv_sha256", "src_model", "tgt_model",
        "pooling", "projection_type", "projection_dim", "proj_dropout", "loss_type",
        "lr", "weight_decay", "temperature", "warmup_ratio", "batch_size",
        "grad_accum_steps", "effective_batch_size", "src_max_len", "tgt_max_len",
        "lowercase", "strip_accents", "remove_punct", "seed", "checkpoint_epoch",
    ]
    config = {key: metadata[key] for key in fields}
    config.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "retrieval": retrieval,
            "use_csls": retrieval == "csls",
            "csls_k": int(csls_k),
        }
    )
    return config


def evaluation_cli_args(config: Dict[str, Any]) -> List[str]:
    args = [
        "--mode", "evaluate",
        "--checkpoint_dir", str(config["checkpoint_dir"]),
        "--pooling", str(config["pooling"]),
        "--projection_type", str(config["projection_type"]),
        "--projection_dim", str(config["projection_dim"]),
        "--proj_dropout", str(config["proj_dropout"]),
        "--loss_type", str(config["loss_type"]),
        "--retrieval", str(config["retrieval"]),
        "--csls_k", str(config["csls_k"]),
        "--src_max_len", str(config["src_max_len"]),
        "--tgt_max_len", str(config["tgt_max_len"]),
        "--batch_size", str(config["batch_size"]),
        "--eval_batch_size", str(max(1, int(config["batch_size"]))),
        "--topk_eval", "10",
        "--eval_ks", "1", "5", "10",
    ]
    for flag, key in (
        ("--lowercase", "lowercase"),
        ("--strip_accents", "strip_accents"),
        ("--remove_punct", "remove_punct"),
    ):
        if config[key]:
            args.append(flag)
    return args


def write_result(
    args: argparse.Namespace,
    output_dir: Path,
    split_name: str,
    configuration_name: str,
    config: Dict[str, Any],
    csv_path: str,
    df: pd.DataFrame,
    raw_src: List[str],
    raw_tgt: List[str],
    norm_src: List[str],
    norm_tgt: List[str],
    metrics: Dict[str, float],
    ranks: np.ndarray,
    topk_idx: np.ndarray,
    scores: np.ndarray,
    save_predictions: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounded: Dict[str, Any] = {key: round(float(value), 4) for key, value in metrics.items()}
    rounded.update(
        {
            "method": "full_encoder_contrastive_finetuning",
            "family": FAMILY,
            "split": split_name,
            "configuration_name": configuration_name,
            "configuration": config,
            "input_csv": csv_path,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": evaluation_cli_args(config),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if save_predictions:
        gold_idx = np.arange(len(df), dtype=np.int64)
        pred_idx = topk_idx[:, 0]
        result = pd.DataFrame(
            {
                "Bahnaric": raw_src,
                "Bahnaric_normalized": norm_src,
                "Predicted_VN": [raw_tgt[index] for index in pred_idx],
                "Gold_VN": raw_tgt,
                "Gold_VN_normalized": norm_tgt,
                "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
                "TopK_indices": ["|".join(map(str, row)) for row in topk_idx],
                "P@1": (pred_idx == gold_idx).astype(np.float32),
                "MRR": np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32),
                "Top1_score": [float(scores[i, pred_idx[i]]) for i in range(len(df))],
                "Gold_score": [float(scores[i, i]) for i in range(len(df))],
            }
        )
        result.to_csv(output_dir / "sentence_predictions.csv", index=False)
    print(json.dumps(rounded, ensure_ascii=False, indent=2))
    return rounded


def evaluate_checkpoint_on_dev(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    checkpoint_epoch: int,
    src_tok,
    tgt_tok,
    src_model,
    tgt_model,
    src_projection,
    tgt_projection,
    metadata: Dict[str, Any],
    device: torch.device,
) -> None:
    (
        df, raw_src, raw_tgt, norm_src, norm_tgt, src_emb, tgt_emb
    ) = encode_eval_pool(
        args, args.dev_csv, src_tok, tgt_tok, src_model, tgt_model,
        src_projection, tgt_projection, device,
    )
    retrievals = ["cosine", "csls"] if args.eval_retrievals == "both" else [args.eval_retrievals]
    for retrieval in retrievals:
        configuration_name = f"{args.configuration_prefix}_{checkpoint_epoch}ep_{retrieval}"
        result_dir = Path(args.dev_output_root) / configuration_name
        metrics, ranks, topk_idx, scores = evaluate_embeddings(
            src_emb, tgt_emb, retrieval, args.topk_eval, args.csls_k,
            [int(value) for value in args.eval_ks],
        )
        config = build_configuration(metadata, checkpoint_dir, retrieval, args.csls_k)
        write_result(
            args, result_dir, "dev", configuration_name, config, args.dev_csv,
            df, raw_src, raw_tgt, norm_src, norm_tgt,
            metrics, ranks, topk_idx, scores, args.save_dev_predictions,
        )


def load_selection(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read selection manifest {path}: {exc}") from exc


def authorize_test(
    selection_manifest: Path,
    configuration_name: str,
    actual_config: Dict[str, Any],
) -> None:
    manifest = load_selection(selection_manifest)
    if FAMILY not in manifest:
        raise ValueError(f"Selection manifest has no {FAMILY!r} entry")
    selected = manifest[FAMILY]
    if selected.get("configuration_name") != configuration_name:
        raise ValueError(
            f"Unauthorized test configuration {configuration_name!r}; development selected "
            f"{selected.get('configuration_name')!r}"
        )
    expected = selected.get("configuration")
    if expected != actual_config:
        raise ValueError(
            "Test arguments/checkpoint do not match the development-selected configuration.\n"
            f"Expected: {json.dumps(expected, sort_keys=True)}\n"
            f"Actual:   {json.dumps(actual_config, sort_keys=True)}"
        )


def evaluate_mode(args: argparse.Namespace) -> None:
    required = {
        "input_csv": args.input_csv,
        "checkpoint_dir": args.checkpoint_dir,
        "output_dir": args.output_dir,
        "configuration_name": args.configuration_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Evaluation mode requires: {missing}")
    if args.split_name == "test" and not args.selection_manifest:
        raise ValueError("Test evaluation requires --selection_manifest")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    (
        src_tok, tgt_tok, src_model, tgt_model,
        src_projection, tgt_projection, metadata,
    ) = load_checkpoint_components(checkpoint_dir, device)

    for key in ("pooling", "projection_type", "projection_dim", "loss_type"):
        cli_value = getattr(args, key)
        checkpoint_value = metadata[key]
        if cli_value != checkpoint_value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: CLI={cli_value!r}, checkpoint={checkpoint_value!r}"
            )

    config = build_configuration(metadata, checkpoint_dir, args.retrieval, args.csls_k)
    if args.split_name == "test":
        authorize_test(Path(args.selection_manifest), args.configuration_name, config)

    (
        df, raw_src, raw_tgt, norm_src, norm_tgt, src_emb, tgt_emb
    ) = encode_eval_pool(
        args, args.input_csv, src_tok, tgt_tok, src_model, tgt_model,
        src_projection, tgt_projection, device,
    )
    metrics, ranks, topk_idx, scores = evaluate_embeddings(
        src_emb, tgt_emb, args.retrieval, args.topk_eval, args.csls_k,
        [int(value) for value in args.eval_ks],
    )
    write_result(
        args, Path(args.output_dir), args.split_name, args.configuration_name,
        config, args.input_csv, df, raw_src, raw_tgt, norm_src, norm_tgt,
        metrics, ranks, topk_idx, scores, True,
    )


def latest_resume_checkpoint(output_dir: Path) -> Optional[Path]:
    latest = output_dir / "checkpoint_latest"
    if (latest / "training_state.pt").is_file():
        return latest
    return None


def build_epoch_loader(
    dataset: BahVnDataset,
    epoch: int,
    start_batch_index: int,
    args: argparse.Namespace,
    src_tok,
    tgt_tok,
    device: torch.device,
) -> DataLoader:
    generator = np.random.default_rng(args.seed + epoch)
    indices = generator.permutation(len(dataset)).tolist()
    offset = int(start_batch_index) * int(args.batch_size)
    remaining = indices[offset:]
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=FixedOrderSampler(remaining),
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_batch(
            batch, src_tok, tgt_tok, args.src_max_len, args.tgt_max_len
        ),
    )


def make_training_state(
    optimizer,
    scheduler,
    scaler,
    next_epoch: int,
    next_batch_index: int,
    global_update_step: int,
) -> Dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "next_epoch": int(next_epoch),
        "next_batch_index": int(next_batch_index),
        "global_update_step": int(global_update_step),
        "rng_state": capture_rng_state(),
    }


def train_mode(args: argparse.Namespace) -> None:
    for name in ("train_csv", "dev_csv", "output_dir", "configuration_prefix"):
        if not getattr(args, name):
            raise ValueError(f"Training mode requires --{name}")
    if Path(args.train_csv).resolve() == Path(args.dev_csv).resolve():
        raise ValueError("train_csv and dev_csv must be different files")
    if args.epochs < max(args.checkpoint_epochs):
        raise ValueError("--epochs must be at least the largest --checkpoint_epochs value")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.dev_output_root).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    resume_dir: Optional[Path] = Path(args.resume_from) if args.resume_from else None
    if resume_dir is None and args.auto_resume:
        resume_dir = latest_resume_checkpoint(output_dir)

    if resume_dir is not None:
        print(f"Resuming from {resume_dir}")
        (
            src_tok, tgt_tok, src_model, tgt_model,
            src_projection, tgt_projection, resume_metadata,
        ) = load_checkpoint_components(resume_dir, device)
        for key in ("pooling", "projection_type", "projection_dim", "loss_type"):
            if getattr(args, key) != resume_metadata[key]:
                raise ValueError(
                    f"Resume checkpoint mismatch for {key}: "
                    f"CLI={getattr(args, key)!r}, checkpoint={resume_metadata[key]!r}"
                )
    else:
        src_tok = AutoTokenizer.from_pretrained(args.src_model)
        tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
        src_model = AutoModel.from_pretrained(args.src_model).to(device)
        tgt_model = AutoModel.from_pretrained(args.tgt_model).to(device)
        src_projection, _ = build_projection(
            args.projection_type, get_hidden_size(src_model),
            args.projection_dim, args.proj_dropout,
        )
        tgt_projection, _ = build_projection(
            args.projection_type, get_hidden_size(tgt_model),
            args.projection_dim, args.proj_dropout,
        )
        src_projection = src_projection.to(device)
        tgt_projection = tgt_projection.to(device)

    train_dataset = BahVnDataset(
        args.train_csv, args.lowercase, args.strip_accents, args.remove_punct
    )
    trainable_params = (
        list(src_model.parameters()) + list(tgt_model.parameters())
        + list(src_projection.parameters()) + list(tgt_projection.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    num_batches = math.ceil(len(train_dataset) / args.batch_size)
    updates_per_epoch = math.ceil(num_batches / max(1, args.grad_accum_steps))
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    next_epoch = 1
    next_batch_index = 0
    global_update_step = 0
    if resume_dir is not None:
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        next_epoch = int(state["next_epoch"])
        next_batch_index = int(state["next_batch_index"])
        global_update_step = int(state["global_update_step"])
        restore_rng_state(state.get("rng_state", {}))

    config_path = output_dir / "training_config.json"
    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    use_amp = bool(args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    start_time = time.monotonic()
    runtime_limit_seconds = (
        float(args.max_runtime_minutes) * 60.0 if args.max_runtime_minutes > 0 else None
    )

    for epoch in range(next_epoch, args.epochs + 1):
        start_batch = next_batch_index if epoch == next_epoch else 0
        loader = build_epoch_loader(
            train_dataset, epoch, start_batch, args, src_tok, tgt_tok, device
        )
        src_model.train()
        tgt_model.train()
        src_projection.train()
        tgt_projection.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed_batches = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for local_index, (src_inputs, tgt_inputs) in enumerate(progress):
            absolute_batch_index = start_batch + local_index + 1
            src_inputs = move_to_device(src_inputs, device)
            tgt_inputs = move_to_device(tgt_inputs, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=use_amp and device.type == "cuda",
            ):
                src_emb = encode_batch(src_model, src_projection, src_inputs, args.pooling)
                tgt_emb = encode_batch(tgt_model, tgt_projection, tgt_inputs, args.pooling)
                loss = contrastive_loss(src_emb, tgt_emb, args.temperature, args.loss_type)
                scaled_loss = loss / max(1, args.grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += float(loss.detach().cpu())
            processed_batches += 1

            should_step = (
                absolute_batch_index % args.grad_accum_steps == 0
                or absolute_batch_index == num_batches
            )
            if should_step:
                if args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update_step += 1

                state = make_training_state(
                    optimizer, scheduler, scaler,
                    next_epoch=epoch,
                    next_batch_index=absolute_batch_index,
                    global_update_step=global_update_step,
                )
                metadata = checkpoint_metadata(args, checkpoint_epoch=epoch)
                if args.save_every_steps > 0 and global_update_step % args.save_every_steps == 0:
                    save_checkpoint_directory(
                        output_dir / "checkpoint_latest",
                        src_model, tgt_model, src_projection, tgt_projection,
                        src_tok, tgt_tok, metadata, state,
                    )

                elapsed = time.monotonic() - start_time
                if runtime_limit_seconds is not None and elapsed >= runtime_limit_seconds:
                    print("Runtime limit reached; saving resumable checkpoint and exiting cleanly.")
                    save_checkpoint_directory(
                        output_dir / "checkpoint_latest",
                        src_model, tgt_model, src_projection, tgt_projection,
                        src_tok, tgt_tok, metadata, state,
                    )
                    return

            progress.set_postfix(
                loss=f"{running_loss / max(1, processed_batches):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        next_batch_index = 0
        end_state = make_training_state(
            optimizer, scheduler, scaler,
            next_epoch=epoch + 1,
            next_batch_index=0,
            global_update_step=global_update_step,
        )
        metadata = checkpoint_metadata(args, checkpoint_epoch=epoch)
        save_checkpoint_directory(
            output_dir / "checkpoint_latest",
            src_model, tgt_model, src_projection, tgt_projection,
            src_tok, tgt_tok, metadata, end_state,
        )

        if epoch in set(args.checkpoint_epochs):
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            save_checkpoint_directory(
                checkpoint_dir,
                src_model, tgt_model, src_projection, tgt_projection,
                src_tok, tgt_tok, metadata, end_state,
            )
            evaluate_checkpoint_on_dev(
                args, checkpoint_dir, epoch,
                src_tok, tgt_tok, src_model, tgt_model,
                src_projection, tgt_projection, metadata, device,
            )

    print("Training and declared development evaluations are complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-encoder contrastive fine-tuning with dev-only configuration selection"
    )
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")

    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--dev_csv", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--split_name", choices=["dev", "test"], default="dev")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dev_output_root", default="results/dev/full_encoder")
    parser.add_argument("--configuration_prefix", default=None)
    parser.add_argument("--configuration_name", default=None)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--checkpoint_dir", default=None)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")
    parser.add_argument(
        "--pooling", choices=["sentence_mean", "token_mean", "cls"],
        default="sentence_mean",
    )
    parser.add_argument(
        "--projection_type", choices=["mlp", "linear", "identity"], default="mlp"
    )
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss_type",
        choices=["symmetric_infonce", "source_to_target_infonce"],
        default="symmetric_infonce",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint_epochs", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)

    parser.add_argument("--eval_retrievals", choices=["cosine", "csls", "both"], default="both")
    parser.add_argument("--retrieval", choices=["cosine", "csls"], default="cosine")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--max_runtime_minutes", type=float, default=0.0)
    parser.add_argument("--save_dev_predictions", action="store_true")

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use only one of --fp16 and --bf16")
    if args.mode == "train":
        train_mode(args)
    else:
        evaluate_mode(args)


if __name__ == "__main__":
    main()
