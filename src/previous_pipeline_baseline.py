# """
# Baseline 6: Previous pipeline / Projection + Procrustes baseline.

# This baseline evaluates the old/current system as a retrieval baseline.

# Recommended main setting:
# - Backbone: XLM-R
# - Epochs: 50
# - Kabsch alignment: 10K lexicon
# - Projection dir: results/models/b2_xlmr_50ep
# - Alignment dir: results/alignment/alignment_B2_xlmr_10K_50ep

# Pipeline:
# 1. Encode Bahnaric and Vietnamese sentences with pretrained encoders.
# 2. Pool encoder token representations.
# 3. Apply trained projection heads: src_proj.pt and tgt_proj.pt.
# 4. Optionally apply Kabsch/Procrustes alignment R.npy, t.npy to Bahnaric embeddings.
# 5. Retrieve nearest Vietnamese sentence using cosine similarity or CSLS.
# 6. Evaluate Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

# This script does NOT train anything.
# It only loads existing projection heads and optional Kabsch alignment.

# Expected files:
# - --proj_dir/src_proj.pt
# - --proj_dir/tgt_proj.pt
# - optionally --alignment_dir/R.npy
# - optionally --alignment_dir/t.npy

# Main XLM-R 50ep + 10K Kabsch example:
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir results/models/b2_xlmr_50ep \
#   --alignment_dir results/alignment/alignment_B2_xlmr_10K_50ep \
#   --src_model xlm-roberta-base \
#   --tgt_model xlm-roberta-base \
#   --pooling token_mean \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_cosine \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# CSLS variant:
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir results/models/b2_xlmr_50ep \
#   --alignment_dir results/alignment/alignment_B2_xlmr_10K_50ep \
#   --src_model xlm-roberta-base \
#   --tgt_model xlm-roberta-base \
#   --pooling token_mean \
#   --use_csls \
#   --csls_k 10 \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_csls \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# No-Kabsch ablation:
# python src/previous_pipeline_baseline.py \
#   --test_csv data/test.csv \
#   --proj_dir results/models/b2_xlmr_50ep \
#   --src_model xlm-roberta-base \
#   --tgt_model xlm-roberta-base \
#   --pooling token_mean \
#   --no_kabsch \
#   --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_cosine \
#   --batch_size 8 \
#   --src_max_len 256 \
#   --tgt_max_len 256 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10
# """

# import argparse
# import json
# import math
# import re
# import unicodedata
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer

# try:
#     from peft import PeftModel
# except Exception:
#     PeftModel = None


# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)


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


# def get_hidden_size(model) -> int:
#     hidden = getattr(model.config, "hidden_size", None)
#     if hidden is None:
#         hidden = getattr(model.config, "dim", None)
#     if hidden is None:
#         raise ValueError("Could not infer hidden size from model config.")
#     return int(hidden)


# def load_projection_head(path: Path, hidden_size: int, device: torch.device) -> Tuple[nn.Module, int]:
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Projection head not found: {path}\n"
#             "Fix: pass a checkpoint subfolder, for example:\n"
#             "  --proj_dir results/models/b2_xlmr_50ep\n"
#             "Do not pass the parent directory results/models unless src_proj.pt and tgt_proj.pt are directly inside it."
#         )

#     sd = torch.load(path, map_location=device)

#     if "net.0.weight" not in sd:
#         raise ValueError(
#             f"{path} does not look like a ProjectionHead state_dict. "
#             "Expected key: net.0.weight"
#         )

#     out_dim = int(sd["net.0.weight"].shape[0])
#     in_dim = int(sd["net.0.weight"].shape[1])

#     if in_dim != hidden_size:
#         raise ValueError(
#             f"Projection input dimension mismatch for {path}. "
#             f"Projection expects in_dim={in_dim}, but encoder hidden_size={hidden_size}. "
#             "Check that --src_model/--tgt_model match the model used to train the projection heads."
#         )

#     head = ProjectionHead(in_dim=hidden_size, out_dim=out_dim).to(device)
#     head.load_state_dict(sd, strict=True)
#     head.eval()
#     return head, out_dim


# def maybe_load_lora(base_model, adapters_dir: Path, required: bool = False):
#     """
#     Load PEFT LoRA adapters.

#     Important:
#     - For the XLM-R B2 checkpoints, projection heads were trained together with LoRA.
#     - If --use_lora is passed but adapters are missing, we should fail loudly.
#       Otherwise the script evaluates vanilla XLM-R with LoRA-trained projection heads,
#       which gives near-random retrieval.
#     """
#     if not adapters_dir.is_dir():
#         if required:
#             raise FileNotFoundError(
#                 f"LoRA adapter directory not found: {adapters_dir}\n"
#                 "This checkpoint was expected to contain PEFT adapters. "
#                 "Check that --proj_dir points to the correct checkpoint folder, e.g. "
#                 "results/models/b2_xlmr_50ep"
#             )
#         return base_model

#     adapter_config = adapters_dir / "adapter_config.json"
#     if required and not adapter_config.exists():
#         raise FileNotFoundError(
#             f"LoRA adapter_config.json not found in: {adapters_dir}\n"
#             "This directory exists, but it does not look like a PEFT LoRA adapter folder."
#         )

#     if PeftModel is None:
#         raise ImportError("peft is not available, but --use_lora was requested.")

#     print(f"Loading LoRA adapter from {adapters_dir}")
#     model = PeftModel.from_pretrained(base_model, str(adapters_dir))
#     model.eval()
#     return model


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


# def build_idf(tokenizer, texts: List[str], max_len: int) -> Dict[int, float]:
#     from collections import Counter

#     df = Counter()
#     n_docs = 0

#     for i in range(0, len(texts), 256):
#         batch = texts[i : i + 256]
#         enc = tokenizer(
#             batch,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )

#         for ids in enc["input_ids"]:
#             n_docs += 1
#             df.update(set(int(x) for x in ids.tolist()))

#     return {tid: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tid, freq in df.items()}


# @torch.no_grad()
# def encode_project_pool(
#     texts: List[str],
#     tokenizer,
#     base_model,
#     proj_head,
#     device: torch.device,
#     max_len: int,
#     batch_size: int,
#     pooling: str,
#     idf_weights: Optional[Dict[int, float]] = None,
#     token_kabsch_R: Optional[np.ndarray] = None,
#     token_kabsch_t: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     """
#     Supported pooling:
#     - sentence_mean:
#         encoder -> mean pool -> projection
#     - token_mean:
#         encoder -> project each token -> mean pool
#     - token_idf:
#         encoder -> project each token -> IDF-weighted pool

#     Optional token-level Kabsch:
#     - If token_kabsch_R and token_kabsch_t are provided, Kabsch is applied
#       after token projection and before pooling:
#         encoder -> project each token -> Kabsch each projected token -> pool
#     - This is intended for source/Bahnaric embeddings only.
#     """
#     base_model.to(device).eval()
#     proj_head.to(device).eval()

#     all_embs = []

#     for start in range(0, len(texts), batch_size):
#         batch = texts[start : start + batch_size]
#         inputs = tokenizer(
#             batch,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         outputs = base_model(**inputs, return_dict=True)
#         last = outputs.last_hidden_state
#         attn = inputs["attention_mask"].float()

#         if pooling == "sentence_mean":
#             mask = attn.unsqueeze(-1)
#             pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#             emb = proj_head(pooled)

#         elif pooling in {"token_mean", "token_idf"}:
#             bsz, seq_len, hidden = last.shape
#             flat = last.reshape(bsz * seq_len, hidden)
#             projected = proj_head(flat).reshape(bsz, seq_len, -1)

#             if token_kabsch_R is not None or token_kabsch_t is not None:
#                 if token_kabsch_R is None or token_kabsch_t is None:
#                     raise ValueError("Both token_kabsch_R and token_kabsch_t must be provided together.")

#                 R_torch = torch.as_tensor(
#                     token_kabsch_R,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )
#                 t_torch = torch.as_tensor(
#                     token_kabsch_t,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )

#                 if R_torch.shape != (projected.shape[-1], projected.shape[-1]):
#                     raise ValueError(
#                         f"token_kabsch_R shape {tuple(R_torch.shape)} does not match "
#                         f"projected token dim {projected.shape[-1]}."
#                     )
#                 if t_torch.shape[0] != projected.shape[-1]:
#                     raise ValueError(
#                         f"token_kabsch_t shape {tuple(t_torch.shape)} does not match "
#                         f"projected token dim {projected.shape[-1]}."
#                     )

#                 # Equivalent to numpy apply_kabsch(x): (R @ x.T).T + t
#                 # For token tensor [batch, seq, dim], this is x @ R.T + t.
#                 projected = torch.matmul(projected, R_torch.T) + t_torch

#             if pooling == "token_idf":
#                 if idf_weights is None:
#                     raise ValueError("pooling=token_idf requires idf_weights.")
#                 token_ids = inputs["input_ids"]
#                 weights = torch.ones_like(token_ids, dtype=torch.float32)
#                 for b in range(bsz):
#                     for t in range(seq_len):
#                         tid = int(token_ids[b, t].item())
#                         weights[b, t] = float(idf_weights.get(tid, 1.0))
#                 weights = weights.to(device) * attn
#             else:
#                 weights = attn

#             weights = weights.unsqueeze(-1)
#             emb = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

#         else:
#             raise ValueError(f"Unknown pooling mode: {pooling}")

#         all_embs.append(emb.detach().cpu().numpy())
#         print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

#     return np.vstack(all_embs).astype(np.float32)


# def apply_kabsch(x: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
#     return (R @ x.T).T + t


# def main(args: argparse.Namespace) -> None:
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

#     print(f"Using device: {device}")
#     print("Baseline: previous_pipeline_projection_procrustes")
#     print(f"Source model: {args.src_model}")
#     print(f"Target model: {args.tgt_model}")
#     print(f"Projection dir: {args.proj_dir}")
#     print(f"Alignment dir: {args.alignment_dir}")
#     print(f"Pooling: {args.pooling}")

#     df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#     raw_bah = df["Bahnaric"].astype(str).tolist()
#     raw_vn = df["Vietnamese"].astype(str).tolist()

#     bah = [
#         normalize_text(
#             x,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for x in raw_bah
#     ]

#     vn = [
#         normalize_text(
#             x,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for x in raw_vn
#     ]

#     src_tok = AutoTokenizer.from_pretrained(args.src_model)
#     tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)

#     src_base = AutoModel.from_pretrained(args.src_model)
#     tgt_base = AutoModel.from_pretrained(args.tgt_model)

#     proj_dir = Path(args.proj_dir)

#     if args.use_lora:
#         print("Loading LoRA adapters because --use_lora was provided...")
#         src_base = maybe_load_lora(src_base, proj_dir / "src_adapters", required=True)
#         tgt_base = maybe_load_lora(tgt_base, proj_dir / "tgt_adapters", required=True)
#     else:
#         print(
#             "WARNING: --use_lora was NOT provided. "
#             "If this checkpoint was trained with LoRA, retrieval may collapse because "
#             "the projection heads expect LoRA-adapted encoder features."
#         )

#     src_hidden = get_hidden_size(src_base)
#     tgt_hidden = get_hidden_size(tgt_base)

#     src_head, src_proj_dim = load_projection_head(proj_dir / "src_proj.pt", src_hidden, device)
#     tgt_head, tgt_proj_dim = load_projection_head(proj_dir / "tgt_proj.pt", tgt_hidden, device)

#     if src_proj_dim != tgt_proj_dim:
#         raise ValueError(f"Source and target projection dims differ: {src_proj_dim} vs {tgt_proj_dim}")

#     proj_dim = src_proj_dim

#     use_kabsch = (not args.no_kabsch) and (args.alignment_dir is not None)
#     R = None
#     t = None

#     if use_kabsch:
#         alignment_dir = Path(args.alignment_dir)
#         R_path = alignment_dir / "R.npy"
#         t_path = alignment_dir / "t.npy"

#         if not R_path.exists() or not t_path.exists():
#             raise FileNotFoundError(
#                 f"Missing Kabsch files in {alignment_dir}. Expected R.npy and t.npy."
#             )

#         R = np.load(R_path)
#         t = np.load(t_path)

#         if R.shape != (proj_dim, proj_dim):
#             raise ValueError(f"R shape {R.shape} does not match projection dim {proj_dim}.")
#         if t.shape[0] != proj_dim:
#             raise ValueError(f"t shape {t.shape} does not match projection dim {proj_dim}.")

#         if args.kabsch_stage == "token" and args.pooling == "sentence_mean":
#             raise ValueError(
#                 "--kabsch_stage token is only valid for token_mean or token_idf pooling. "
#                 "sentence_mean pools before projection, so there are no projected token vectors to align."
#             )

#         print(f"[Kabsch] Loaded R.npy and t.npy from {alignment_dir}")
#         print(f"[Kabsch] Stage: {args.kabsch_stage}")
#     else:
#         print("[Kabsch] Disabled or no alignment_dir provided.")

#     src_idf = None
#     tgt_idf = None
#     if args.pooling == "token_idf":
#         print("Building source IDF weights...")
#         src_idf = build_idf(src_tok, bah, max_len=args.src_max_len)
#         print("Building target IDF weights...")
#         tgt_idf = build_idf(tgt_tok, vn, max_len=args.tgt_max_len)

#     print("[1/3] Encoding Bahnaric queries...")
#     bah_emb = encode_project_pool(
#         texts=bah,
#         tokenizer=src_tok,
#         base_model=src_base,
#         proj_head=src_head,
#         device=device,
#         max_len=args.src_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=src_idf,
#         token_kabsch_R=R if use_kabsch and args.kabsch_stage == "token" else None,
#         token_kabsch_t=t if use_kabsch and args.kabsch_stage == "token" else None,
#     )

#     print("[2/3] Encoding Vietnamese candidates...")
#     vn_emb = encode_project_pool(
#         texts=vn,
#         tokenizer=tgt_tok,
#         base_model=tgt_base,
#         proj_head=tgt_head,
#         device=device,
#         max_len=args.tgt_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=tgt_idf,
#         token_kabsch_R=None,
#         token_kabsch_t=None,
#     )

#     if use_kabsch and args.kabsch_stage == "sentence":
#         print("[Kabsch] Applying R.npy and t.npy to Bahnaric sentence embeddings...")
#         bah_retrieval_emb = apply_kabsch(bah_emb, R, t)
#         kabsch_used = True
#     elif use_kabsch and args.kabsch_stage == "token":
#         print("[Kabsch] Already applied to Bahnaric projected token embeddings before pooling.")
#         bah_retrieval_emb = bah_emb
#         kabsch_used = True
#     else:
#         print("[Kabsch] Skipped. Using projected embeddings directly.")
#         bah_retrieval_emb = bah_emb
#         kabsch_used = False

#     print(f"Bahnaric embedding shape: {bah_retrieval_emb.shape}")
#     print(f"Vietnamese embedding shape: {vn_emb.shape}")

#     print("[3/3] Retrieving candidates...")
#     if args.use_csls:
#         topk_idx, scores = csls_topk(
#             query=bah_retrieval_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#         retrieval = "csls"
#     else:
#         topk_idx, scores = cosine_topk(
#             query=bah_retrieval_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#         )
#         retrieval = "cosine"

#     gold_idx = np.arange(len(df), dtype=np.int64)
#     eval_ks = [int(k) for k in args.eval_ks]

#     metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

#     pred_top1_idx = topk_idx[:, 0]
#     preds = [raw_vn[i] for i in pred_top1_idx]
#     topk_preds = ["|".join([raw_vn[j] for j in row]) for row in topk_idx]

#     is_top1 = (pred_top1_idx == gold_idx).astype(np.float32)
#     rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)
#     gold_rank = [None if not np.isfinite(r) else int(r) for r in ranks.tolist()]

#     out = {
#         "Bahnaric": raw_bah,
#         "Bahnaric_normalized": bah,
#         "Predicted_VN": preds,
#         "Gold_VN": raw_vn,
#         "Gold_VN_normalized": vn,
#         "Gold_rank": gold_rank,
#         "TopK_Preds": topk_preds,
#         "P@1": is_top1.tolist(),
#         "MRR": rr.tolist(),
#         "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
#         "Gold_score": [float(scores[i, i]) for i in range(len(df))],
#     }

#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         out[f"Hit@{k_eff}"] = (ranks <= float(k_eff)).astype(np.float32).tolist()

#     df_out = pd.DataFrame(out)
#     df_out = make_bucket_columns(df_out)
#     df_out.to_csv(output_dir / "sentence_predictions.csv", index=False)

#     save_bucket_metrics(df_out, output_dir, eval_ks)

#     rounded_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
#     rounded_metrics.update(
#         {
#             "method": "previous_pipeline_projection_procrustes",
#             "pipeline": "encoder_pool_projection_optional_kabsch_retrieval",
#             "test_csv": args.test_csv,
#             "proj_dir": args.proj_dir,
#             "alignment_dir": args.alignment_dir,
#             "src_model": args.src_model,
#             "tgt_model": args.tgt_model,
#             "embedding_dim": int(proj_dim),
#             "pooling": args.pooling,
#             "retrieval": retrieval,
#             "num_queries": int(len(df)),
#             "candidate_pool_size": int(len(df)),
#             "topk_eval": int(args.topk_eval),
#             "eval_ks": eval_ks,
#             "src_max_len": int(args.src_max_len),
#             "tgt_max_len": int(args.tgt_max_len),
#             "batch_size": int(args.batch_size),
#             "kabsch_used": bool(kabsch_used),
#             "kabsch_stage": args.kabsch_stage,
#             "use_csls": bool(args.use_csls),
#             "csls_k": int(args.csls_k),
#             "use_lora": bool(args.use_lora),
#             "strip_accents": bool(args.strip_accents),
#             "lowercase": bool(args.lowercase),
#             "remove_punct": bool(args.remove_punct),
#         }
#     )

#     with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
#         json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
#     print(f"Saved metrics to {output_dir / 'metrics.json'}")
#     print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Previous pipeline baseline: encoder + projection + optional Kabsch + cosine/CSLS retrieval"
#     )

#     parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
#     parser.add_argument("--proj_dir", required=True, help="Directory with src_proj.pt and tgt_proj.pt")
#     parser.add_argument("--alignment_dir", default=None, help="Directory with R.npy and t.npy")
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--src_model", default="xlm-roberta-base")
#     parser.add_argument("--tgt_model", default="xlm-roberta-base")

#     parser.add_argument(
#         "--pooling",
#         choices=["sentence_mean", "token_mean", "token_idf"],
#         default="token_mean",
#         help=(
#             "sentence_mean: encoder -> mean pool -> projection; "
#             "token_mean: encoder -> projection per token -> mean pool; "
#             "token_idf: encoder -> projection per token -> IDF-weighted pool"
#         ),
#     )

#     parser.add_argument("--src_max_len", type=int, default=256)
#     parser.add_argument("--tgt_max_len", type=int, default=256)
#     parser.add_argument("--batch_size", type=int, default=8)

#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument("--no_kabsch", action="store_true", help="Do not apply R.npy/t.npy")
#     parser.add_argument(
#         "--kabsch_stage",
#         choices=["sentence", "token"],
#         default="sentence",
#         help=(
#             "Where to apply Kabsch when --alignment_dir is provided. "
#             "sentence: apply to final Bahnaric sentence embeddings after pooling. "
#             "token: apply to projected Bahnaric token embeddings before pooling; "
#             "valid only for token_mean and token_idf."
#         ),
#     )
#     parser.add_argument("--use_csls", action="store_true", help="Use CSLS instead of cosine")
#     parser.add_argument("--csls_k", type=int, default=10)

#     parser.add_argument("--use_lora", action="store_true", help="Load PEFT LoRA adapters from proj_dir/src_adapters and proj_dir/tgt_adapters")

#     parser.add_argument("--lowercase", action="store_true")
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")

#     parser.add_argument("--no_cuda", action="store_true")

#     args = parser.parse_args()
#     main(args)

########################################
# #!/usr/bin/env python3
# """Reviewer-compliant XLM-R LoRA projection retrieval evaluator.

# Every XLM-R LoRA/projection/Kabsch/retrieval variant must first be evaluated on
# ``data/dev.csv``. Only the exact configuration recorded in
# ``results/dev_selection/selected_configs.json`` may be evaluated on the held-out
# ``data/test.csv``.

# This file evaluates an existing LoRA + projection checkpoint. Training is
# handled by ``src/lora_projection_generalization_train_eval.py``. The evaluator
# supports:

# * sentence-mean, token-mean, and token-IDF pooling;
# * no Kabsch, sentence-level Kabsch, and token-level Kabsch;
# * cosine and CSLS retrieval;
# * resumable embedding caching; and
# * exact test-time authorization against the development-selection manifest.
# """

# from __future__ import annotations

# import argparse
# import hashlib
# import json
# import math
# import os
# import re
# import unicodedata
# from collections import Counter
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer

# try:
#     from peft import PeftModel
# except Exception:
#     PeftModel = None

# FAMILY = "xlmr_lora_projection"
# EVALUATOR_SCRIPT = "src/previous_pipeline_baseline.py"
# REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# def normalize_text(
#     text: str,
#     lowercase: bool = False,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
# ) -> str:
#     text = unicodedata.normalize("NFC", str(text))
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

#     return re.sub(r"\s+", " ", text).strip()


# def read_parallel_csv(path: Path) -> pd.DataFrame:
#     if not path.is_file():
#         raise FileNotFoundError(f"CSV not found: {path}")
#     df = pd.read_csv(path)
#     missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
#     if missing:
#         raise ValueError(f"{path} is missing required columns: {missing}")
#     return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


# def sha256_file(path: Path) -> str:
#     digest = hashlib.sha256()
#     with path.open("rb") as handle:
#         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
#             digest.update(chunk)
#     return digest.hexdigest()


# def sha256_optional_file(path: Optional[Path]) -> Optional[str]:
#     return sha256_file(path) if path is not None and path.is_file() else None


# def canonical_json(value: Any) -> str:
#     return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# def directory_artifact_fingerprint(paths: List[Path]) -> str:
#     records: List[Dict[str, Any]] = []
#     for path in paths:
#         if path.is_file():
#             records.append(
#                 {
#                     "path": str(path),
#                     "size": int(path.stat().st_size),
#                     "sha256": sha256_file(path),
#                 }
#             )
#         elif path.is_dir():
#             for child in sorted(p for p in path.rglob("*") if p.is_file()):
#                 records.append(
#                     {
#                         "path": str(child),
#                         "size": int(child.stat().st_size),
#                         "sha256": sha256_file(child),
#                     }
#                 )
#         else:
#             records.append({"path": str(path), "missing": True})
#     return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


# def get_hidden_size(model: nn.Module) -> int:
#     hidden = getattr(model.config, "hidden_size", None)
#     if hidden is None:
#         hidden = getattr(model.config, "dim", None)
#     if hidden is None:
#         raise ValueError("Could not infer hidden size from model config")
#     return int(hidden)


# def load_projection_head(
#     path: Path,
#     hidden_size: int,
#     device: torch.device,
# ) -> Tuple[nn.Module, int]:
#     if not path.is_file():
#         raise FileNotFoundError(f"Projection head not found: {path}")

#     state_dict = torch.load(path, map_location=device)
#     if "net.0.weight" not in state_dict:
#         raise ValueError(
#             f"{path} is not a ProjectionHead state_dict: missing net.0.weight"
#         )

#     out_dim = int(state_dict["net.0.weight"].shape[0])
#     in_dim = int(state_dict["net.0.weight"].shape[1])
#     if in_dim != hidden_size:
#         raise ValueError(
#             f"Projection input dimension mismatch for {path}: "
#             f"checkpoint={in_dim}, encoder={hidden_size}"
#         )

#     head = ProjectionHead(hidden_size, out_dim).to(device)
#     head.load_state_dict(state_dict, strict=True)
#     head.eval()
#     return head, out_dim


# def maybe_load_lora(base_model: nn.Module, adapters_dir: Path, required: bool) -> nn.Module:
#     if not required:
#         return base_model
#     if not adapters_dir.is_dir():
#         raise FileNotFoundError(f"LoRA adapter directory not found: {adapters_dir}")
#     if not (adapters_dir / "adapter_config.json").is_file():
#         raise FileNotFoundError(
#             f"LoRA adapter_config.json not found in {adapters_dir}"
#         )
#     if PeftModel is None:
#         raise ImportError("peft is required because --use_lora was requested")
#     print(f"Loading LoRA adapter from {adapters_dir}")
#     model = PeftModel.from_pretrained(base_model, str(adapters_dir))
#     model.eval()
#     return model


# def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return x / norms


# def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
#     k = int(max(1, min(topk, scores.shape[1])))
#     if k == scores.shape[1]:
#         return np.argsort(-scores, axis=1)
#     unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
#     selected = np.take_along_axis(scores, unsorted, axis=1)
#     order = np.argsort(-selected, axis=1)
#     return np.take_along_axis(unsorted, order, axis=1)


# def cosine_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     scores = l2_normalize_rows(query) @ l2_normalize_rows(index).T
#     return topk_from_scores(scores, topk), scores


# def csls_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
#     csls_k: int = 10,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)
#     cosine = q @ z.T

#     if q.shape[0] <= 1 or z.shape[0] <= 1:
#         return topk_from_scores(cosine, topk), cosine

#     k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))
#     rq = np.partition(cosine, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
#     rz = np.partition(cosine, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)
#     scores = 2.0 * cosine - rq[:, None] - rz[None, :]
#     return topk_from_scores(scores, topk), scores


# def ranking_metrics(
#     topk_idx: np.ndarray,
#     gold_idx: np.ndarray,
#     eval_ks: List[int],
# ) -> Tuple[Dict[str, float], np.ndarray]:
#     ranks = np.full(topk_idx.shape[0], np.inf, dtype=np.float64)
#     for row in range(topk_idx.shape[0]):
#         hits = np.where(topk_idx[row] == gold_idx[row])[0]
#         if len(hits):
#             ranks[row] = float(hits[0] + 1)

#     metrics: Dict[str, float] = {
#         "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
#         "Top1_acc": float(np.mean(ranks == 1.0)),
#     }
#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         hit = float(np.mean(ranks <= k_eff))
#         metrics[f"Hit@{k_eff}"] = hit
#         metrics[f"Recall@{k_eff}"] = hit
#         metrics[f"Precision@{k_eff}"] = hit / float(k_eff)
#     return metrics, ranks


# def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
#     result = df_out.copy()
#     result["Bahnaric_len_chars"] = result["Bahnaric"].astype(str).str.len()
#     result["Vietnamese_len_chars"] = result["Gold_VN"].astype(str).str.len()
#     result["Vietnamese_len_words"] = (
#         result["Gold_VN"].astype(str).str.split().map(len)
#     )

#     def vn_len_bin(n: int) -> str:
#         if n <= 5:
#             return "1-5"
#         if n <= 15:
#             return "6-15"
#         if n <= 30:
#             return "16-30"
#         return ">30"

#     result["VN_len_bin"] = result["Vietnamese_len_words"].map(vn_len_bin)
#     return result


# def save_bucket_metrics(
#     df_out: pd.DataFrame,
#     output_dir: Path,
#     eval_ks: List[int],
# ) -> None:
#     metric_cols = ["P@1", "MRR"] + [
#         f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns
#     ]
#     available = [column for column in metric_cols if column in df_out.columns]
#     grouped = df_out.groupby("VN_len_bin", dropna=False)
#     bucket_df = grouped[available].mean().reset_index()
#     bucket_df["count"] = grouped.size().values
#     for column in available:
#         bucket_df[column] = bucket_df[column].astype(float).round(4)
#     bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


# def build_idf(tokenizer: Any, texts: List[str], max_len: int) -> Dict[int, float]:
#     document_frequency: Counter[int] = Counter()
#     n_docs = 0
#     for start in range(0, len(texts), 256):
#         encoded = tokenizer(
#             texts[start : start + 256],
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         attention_mask = encoded["attention_mask"]
#         for token_ids, mask in zip(encoded["input_ids"], attention_mask):
#             active_ids = token_ids[mask.bool()].tolist()
#             document_frequency.update(set(int(token_id) for token_id in active_ids))
#             n_docs += 1
#     return {
#         token_id: math.log((n_docs + 1) / (frequency + 1)) + 1.0
#         for token_id, frequency in document_frequency.items()
#     }


# @torch.inference_mode()
# def encode_project_pool(
#     texts: List[str],
#     tokenizer: Any,
#     base_model: nn.Module,
#     projection_head: nn.Module,
#     device: torch.device,
#     max_len: int,
#     batch_size: int,
#     pooling: str,
#     idf_weights: Optional[Dict[int, float]] = None,
#     token_kabsch_r: Optional[np.ndarray] = None,
#     token_kabsch_t: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     base_model.to(device).eval()
#     projection_head.to(device).eval()
#     all_embeddings: List[np.ndarray] = []

#     for start in range(0, len(texts), batch_size):
#         batch = texts[start : start + batch_size]
#         inputs = tokenizer(
#             batch,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = base_model(**inputs, return_dict=True)
#         hidden = outputs.last_hidden_state
#         attention = inputs["attention_mask"].float()

#         if pooling == "sentence_mean":
#             mask = attention.unsqueeze(-1)
#             pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#             embedding = projection_head(pooled)
#         elif pooling in {"token_mean", "token_idf"}:
#             batch_size_actual, sequence_length, hidden_size = hidden.shape
#             projected = projection_head(
#                 hidden.reshape(batch_size_actual * sequence_length, hidden_size)
#             ).reshape(batch_size_actual, sequence_length, -1)

#             if token_kabsch_r is not None or token_kabsch_t is not None:
#                 if token_kabsch_r is None or token_kabsch_t is None:
#                     raise ValueError(
#                         "token_kabsch_r and token_kabsch_t must be provided together"
#                     )
#                 r_tensor = torch.as_tensor(
#                     token_kabsch_r,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )
#                 t_tensor = torch.as_tensor(
#                     token_kabsch_t,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )
#                 projected = torch.matmul(projected, r_tensor.T) + t_tensor

#             if pooling == "token_idf":
#                 if idf_weights is None:
#                     raise ValueError("pooling=token_idf requires IDF weights")
#                 token_ids = inputs["input_ids"]
#                 weights = torch.ones_like(token_ids, dtype=torch.float32)
#                 for batch_index in range(batch_size_actual):
#                     for token_index in range(sequence_length):
#                         token_id = int(token_ids[batch_index, token_index].item())
#                         weights[batch_index, token_index] = float(
#                             idf_weights.get(token_id, 1.0)
#                         )
#                 weights = weights.to(device) * attention
#             else:
#                 weights = attention

#             expanded_weights = weights.unsqueeze(-1)
#             embedding = (projected * expanded_weights).sum(dim=1) / (
#                 expanded_weights.sum(dim=1).clamp(min=1e-9)
#             )
#         else:
#             raise ValueError(f"Unknown pooling mode: {pooling}")

#         all_embeddings.append(embedding.detach().cpu().numpy())
#         print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

#     return np.vstack(all_embeddings).astype(np.float32)


# def apply_kabsch(x: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
#     return (r @ x.T).T + t


# def load_checkpoint_manifest(args: argparse.Namespace, proj_dir: Path) -> Dict[str, Any]:
#     manifest_path: Optional[Path]
#     if args.checkpoint_manifest:
#         manifest_path = Path(args.checkpoint_manifest)
#     else:
#         candidate = proj_dir / "training_manifest.json"
#         manifest_path = candidate if candidate.is_file() else None

#     if manifest_path is None:
#         if args.allow_unverified_checkpoint:
#             return {
#                 "verified": False,
#                 "warning": "checkpoint training manifest not provided",
#             }
#         raise FileNotFoundError(
#             "Reviewer-compliant evaluation requires a checkpoint training manifest. "
#             "Retrain with src/lora_projection_generalization_train_eval.py or pass "
#             "--allow_unverified_checkpoint only for legacy reproduction, not for the "
#             "reviewer-response result."
#         )

#     if not manifest_path.is_file():
#         raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
#     manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#     if manifest.get("training_policy") != "train_fit_only_no_dev_refit":
#         if not args.allow_unverified_checkpoint:
#             raise ValueError(
#                 f"Checkpoint manifest {manifest_path} does not declare "
#                 "training_policy=train_fit_only_no_dev_refit"
#             )
#     return {
#         "verified": manifest.get("training_policy") == "train_fit_only_no_dev_refit",
#         "manifest_path": str(manifest_path),
#         "manifest_sha256": sha256_file(manifest_path),
#         "training_policy": manifest.get("training_policy"),
#         "train_csv": manifest.get("train_csv"),
#         "train_csv_sha256": manifest.get("train_csv_sha256"),
#         "num_train_pairs": manifest.get("num_train_pairs"),
#         "train_sample_size": manifest.get("train_sample_size"),
#         "epochs": manifest.get("epochs"),
#         "seed": manifest.get("seed"),
#         "projection_dim": manifest.get("projection_dim"),
#         "pooling": manifest.get("pooling"),
#         "lora_r": manifest.get("lora_r"),
#         "lora_alpha": manifest.get("lora_alpha"),
#         "lora_dropout": manifest.get("lora_dropout"),
#         "learning_rate": manifest.get("learning_rate"),
#         "temperature": manifest.get("temperature"),
#     }


# def load_alignment(
#     args: argparse.Namespace,
#     projection_dim: int,
# ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
#     if args.no_kabsch or args.alignment_dir is None:
#         return None, None, {
#             "used": False,
#             "stage": "none",
#             "alignment_dir": None,
#             "artifact_fingerprint": None,
#         }

#     alignment_dir = Path(args.alignment_dir)
#     r_path = alignment_dir / "R.npy"
#     t_path = alignment_dir / "t.npy"
#     if not r_path.is_file() or not t_path.is_file():
#         raise FileNotFoundError(
#             f"Missing R.npy or t.npy under alignment directory {alignment_dir}"
#         )

#     r = np.load(r_path)
#     t = np.load(t_path)
#     if r.shape != (projection_dim, projection_dim):
#         raise ValueError(
#             f"R shape {r.shape} does not match projection dim {projection_dim}"
#         )
#     if t.shape not in {(projection_dim,), (1, projection_dim)}:
#         raise ValueError(
#             f"t shape {t.shape} does not match projection dim {projection_dim}"
#         )
#     t = t.reshape(-1)

#     if args.kabsch_stage == "token" and args.pooling == "sentence_mean":
#         raise ValueError(
#             "Token-level Kabsch is valid only for token_mean or token_idf pooling"
#         )

#     manifest_path = (
#         Path(args.alignment_manifest)
#         if args.alignment_manifest
#         else alignment_dir / "alignment_manifest.json"
#     )
#     provenance: Dict[str, Any] = {
#         "used": True,
#         "stage": args.kabsch_stage,
#         "alignment_dir": str(alignment_dir),
#         "artifact_fingerprint": directory_artifact_fingerprint([r_path, t_path]),
#         "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
#         "manifest_sha256": sha256_optional_file(manifest_path),
#         "provenance_verified": False,
#     }
#     if manifest_path.is_file():
#         manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#         provenance.update(
#             {
#                 "provenance_verified": bool(
#                     manifest.get("fit_policy")
#                     in {
#                         "train_fit_only",
#                         "independent_training_lexicon_no_dev_or_test",
#                     }
#                 ),
#                 "fit_policy": manifest.get("fit_policy"),
#                 "fit_csv": manifest.get("fit_csv"),
#                 "fit_csv_sha256": manifest.get("fit_csv_sha256"),
#                 "alignment_sample_size": manifest.get("alignment_sample_size"),
#             }
#         )
#     elif args.require_alignment_manifest:
#         raise FileNotFoundError(
#             f"Alignment manifest required but not found: {manifest_path}"
#         )

#     return r.astype(np.float32), t.astype(np.float32), provenance


# def build_configuration(
#     args: argparse.Namespace,
#     checkpoint_info: Dict[str, Any],
#     checkpoint_fingerprint: str,
#     alignment_info: Dict[str, Any],
#     idf_info: Dict[str, Any],
# ) -> Dict[str, Any]:
#     return {
#         "src_model": args.src_model,
#         "tgt_model": args.tgt_model,
#         "proj_dir": args.proj_dir,
#         "checkpoint_fingerprint": checkpoint_fingerprint,
#         "checkpoint_training": checkpoint_info,
#         "pooling": args.pooling,
#         "retrieval": "csls" if args.use_csls else "cosine",
#         "kabsch_used": bool(alignment_info.get("used")),
#         "kabsch_stage": alignment_info.get("stage"),
#         "alignment": alignment_info,
#         "use_lora": bool(args.use_lora),
#         "lowercase": bool(args.lowercase),
#         "strip_accents": bool(args.strip_accents),
#         "remove_punct": bool(args.remove_punct),
#         "idf": idf_info,
#         "src_max_len": int(args.src_max_len),
#         "tgt_max_len": int(args.tgt_max_len),
#         "batch_size": int(args.batch_size),
#         "topk_eval": int(args.topk_eval),
#         "eval_ks": [int(k) for k in args.eval_ks],
#         "use_csls": bool(args.use_csls),
#         "csls_k": int(args.csls_k),
#     }


# def build_evaluation_cli_args(args: argparse.Namespace) -> List[str]:
#     cli = [
#         "--proj_dir",
#         args.proj_dir,
#         "--src_model",
#         args.src_model,
#         "--tgt_model",
#         args.tgt_model,
#         "--pooling",
#         args.pooling,
#         "--batch_size",
#         str(args.batch_size),
#         "--src_max_len",
#         str(args.src_max_len),
#         "--tgt_max_len",
#         str(args.tgt_max_len),
#         "--topk_eval",
#         str(args.topk_eval),
#         "--eval_ks",
#         *[str(k) for k in args.eval_ks],
#         "--csls_k",
#         str(args.csls_k),
#     ]
#     if args.checkpoint_manifest:
#         cli.extend(["--checkpoint_manifest", args.checkpoint_manifest])
#     if args.allow_unverified_checkpoint:
#         cli.append("--allow_unverified_checkpoint")
#     if args.alignment_dir:
#         cli.extend(["--alignment_dir", args.alignment_dir])
#     if args.alignment_manifest:
#         cli.extend(["--alignment_manifest", args.alignment_manifest])
#     if args.require_alignment_manifest:
#         cli.append("--require_alignment_manifest")
#     if args.no_kabsch:
#         cli.append("--no_kabsch")
#     else:
#         cli.extend(["--kabsch_stage", args.kabsch_stage])
#     if args.use_lora:
#         cli.append("--use_lora")
#     if args.use_csls:
#         cli.append("--use_csls")
#     if args.lowercase:
#         cli.append("--lowercase")
#     if args.strip_accents:
#         cli.append("--strip_accents")
#     if args.remove_punct:
#         cli.append("--remove_punct")
#     if args.idf_csv:
#         cli.extend(["--idf_csv", args.idf_csv])
#     if args.embedding_cache_dir:
#         cli.extend(["--embedding_cache_dir", args.embedding_cache_dir])
#     if args.no_cuda:
#         cli.append("--no_cuda")
#     if args.no_mps:
#         cli.append("--no_mps")
#     return cli


# def validate_test_authorization(
#     args: argparse.Namespace,
#     configuration: Dict[str, Any],
# ) -> None:
#     if args.split_name != "test":
#         return
#     if not args.selection_manifest:
#         raise ValueError(
#             "Test evaluation requires --selection_manifest. Select on dev first."
#         )
#     manifest_path = Path(args.selection_manifest)
#     if not manifest_path.is_file():
#         raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
#     manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#     if FAMILY not in manifest:
#         raise ValueError(f"Selection manifest has no {FAMILY!r} entry")
#     selected = manifest[FAMILY]
#     expected_name = selected.get("configuration_name")
#     expected_configuration = selected.get("configuration")
#     if args.configuration_name != expected_name:
#         raise ValueError(
#             f"Unauthorized test configuration {args.configuration_name!r}; "
#             f"development selected {expected_name!r}"
#         )
#     if canonical_json(configuration) != canonical_json(expected_configuration):
#         raise ValueError(
#             "Test arguments or model artifacts do not exactly match the "
#             "development-selected XLM-R LoRA configuration"
#         )


# def cache_key(
#     input_path: Path,
#     configuration: Dict[str, Any],
# ) -> str:
#     cache_configuration = dict(configuration)
#     # Retrieval and sentence-level Kabsch do not change the pre-retrieval base
#     # embeddings, so exclude them to maximize reuse. Token-level Kabsch stays in
#     # the key because it changes source embeddings before pooling.
#     cache_configuration.pop("retrieval", None)
#     cache_configuration.pop("use_csls", None)
#     cache_configuration.pop("csls_k", None)
#     alignment = dict(cache_configuration.get("alignment", {}))
#     if cache_configuration.get("kabsch_stage") == "sentence":
#         alignment = {
#             "used": False,
#             "stage": "none_for_base_embedding_cache",
#         }
#         cache_configuration["kabsch_used"] = False
#         cache_configuration["kabsch_stage"] = "none_for_base_embedding_cache"
#     cache_configuration["alignment"] = alignment
#     payload = {
#         "input_sha256": sha256_file(input_path),
#         "configuration": cache_configuration,
#     }
#     return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:28]


# def load_or_encode_embeddings(
#     args: argparse.Namespace,
#     input_path: Path,
#     configuration: Dict[str, Any],
#     source_texts: List[str],
#     target_texts: List[str],
#     source_tokenizer: Any,
#     target_tokenizer: Any,
#     source_model: nn.Module,
#     target_model: nn.Module,
#     source_head: nn.Module,
#     target_head: nn.Module,
#     device: torch.device,
#     source_idf: Optional[Dict[int, float]],
#     target_idf: Optional[Dict[int, float]],
#     r: Optional[np.ndarray],
#     t: Optional[np.ndarray],
# ) -> Tuple[np.ndarray, np.ndarray, str]:
#     cache_path: Optional[Path] = None
#     if args.embedding_cache_dir:
#         cache_dir = Path(args.embedding_cache_dir)
#         cache_dir.mkdir(parents=True, exist_ok=True)
#         cache_path = cache_dir / f"{cache_key(input_path, configuration)}.npz"
#         if cache_path.is_file():
#             try:
#                 with np.load(cache_path) as data:
#                     source_embeddings = np.asarray(data["source"], dtype=np.float32)
#                     target_embeddings = np.asarray(data["target"], dtype=np.float32)
#                 if (
#                     source_embeddings.shape[0] == len(source_texts)
#                     and target_embeddings.shape[0] == len(target_texts)
#                 ):
#                     print(f"Loaded embedding cache: {cache_path}")
#                     return source_embeddings, target_embeddings, str(cache_path)
#             except (OSError, ValueError, KeyError) as exc:
#                 print(f"Ignoring invalid cache {cache_path}: {exc}")

#     token_level_kabsch = (
#         configuration["kabsch_used"] and configuration["kabsch_stage"] == "token"
#     )
#     print("[1/3] Encoding Bahnaric queries...")
#     source_embeddings = encode_project_pool(
#         texts=source_texts,
#         tokenizer=source_tokenizer,
#         base_model=source_model,
#         projection_head=source_head,
#         device=device,
#         max_len=args.src_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=source_idf,
#         token_kabsch_r=r if token_level_kabsch else None,
#         token_kabsch_t=t if token_level_kabsch else None,
#     )
#     print("[2/3] Encoding Vietnamese candidates...")
#     target_embeddings = encode_project_pool(
#         texts=target_texts,
#         tokenizer=target_tokenizer,
#         base_model=target_model,
#         projection_head=target_head,
#         device=device,
#         max_len=args.tgt_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=target_idf,
#     )

#     if cache_path is not None:
#         temporary = cache_path.with_suffix(".tmp.npz")
#         np.savez_compressed(
#             temporary,
#             source=source_embeddings,
#             target=target_embeddings,
#         )
#         os.replace(temporary, cache_path)
#         print(f"Saved embedding cache: {cache_path}")

#     return (
#         source_embeddings,
#         target_embeddings,
#         str(cache_path) if cache_path is not None else "",
#     )


# def main(args: argparse.Namespace) -> None:
#     input_path = Path(args.input_csv)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if args.split_name == "test" and "test" not in input_path.name.lower():
#         print(
#             "WARNING: split_name=test but input filename does not contain 'test'. "
#             "Authorization still relies on the selection manifest."
#         )

#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.no_mps:
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     proj_dir = Path(args.proj_dir)
#     required_projection_paths = [
#         proj_dir / "src_proj.pt",
#         proj_dir / "tgt_proj.pt",
#     ]
#     if args.use_lora:
#         required_projection_paths.extend(
#             [proj_dir / "src_adapters", proj_dir / "tgt_adapters"]
#         )
#     checkpoint_fingerprint = directory_artifact_fingerprint(required_projection_paths)
#     checkpoint_info = load_checkpoint_manifest(args, proj_dir)

#     dataframe = read_parallel_csv(input_path)
#     raw_source = dataframe["Bahnaric"].astype(str).tolist()
#     raw_target = dataframe["Vietnamese"].astype(str).tolist()
#     source_texts = [
#         normalize_text(
#             text,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_source
#     ]
#     target_texts = [
#         normalize_text(
#             text,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_target
#     ]

#     source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
#     target_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
#     source_model = AutoModel.from_pretrained(args.src_model)
#     target_model = AutoModel.from_pretrained(args.tgt_model)
#     source_model = maybe_load_lora(
#         source_model,
#         proj_dir / "src_adapters",
#         required=args.use_lora,
#     )
#     target_model = maybe_load_lora(
#         target_model,
#         proj_dir / "tgt_adapters",
#         required=args.use_lora,
#     )

#     source_head, source_projection_dim = load_projection_head(
#         proj_dir / "src_proj.pt", get_hidden_size(source_model), device
#     )
#     target_head, target_projection_dim = load_projection_head(
#         proj_dir / "tgt_proj.pt", get_hidden_size(target_model), device
#     )
#     if source_projection_dim != target_projection_dim:
#         raise ValueError(
#             f"Projection dimensions differ: {source_projection_dim} vs "
#             f"{target_projection_dim}"
#         )

#     r, t, alignment_info = load_alignment(args, source_projection_dim)

#     source_idf: Optional[Dict[int, float]] = None
#     target_idf: Optional[Dict[int, float]] = None
#     idf_info: Dict[str, Any] = {
#         "used": args.pooling == "token_idf",
#         "idf_csv": None,
#         "idf_csv_sha256": None,
#         "policy": "not_used",
#     }
#     if args.pooling == "token_idf":
#         if not args.idf_csv:
#             raise ValueError(
#                 "pooling=token_idf requires --idf_csv. Use data/train_fit.csv so "
#                 "IDF statistics are fixed before development and test evaluation."
#             )
#         idf_path = Path(args.idf_csv)
#         idf_df = read_parallel_csv(idf_path)
#         idf_source = [
#             normalize_text(
#                 text,
#                 lowercase=args.lowercase,
#                 strip_accents=args.strip_accents,
#                 remove_punct=args.remove_punct,
#             )
#             for text in idf_df["Bahnaric"].astype(str).tolist()
#         ]
#         idf_target = [
#             normalize_text(
#                 text,
#                 lowercase=args.lowercase,
#                 strip_accents=args.strip_accents,
#                 remove_punct=args.remove_punct,
#             )
#             for text in idf_df["Vietnamese"].astype(str).tolist()
#         ]
#         print("Building source IDF weights from fixed training data...")
#         source_idf = build_idf(source_tokenizer, idf_source, args.src_max_len)
#         print("Building target IDF weights from fixed training data...")
#         target_idf = build_idf(target_tokenizer, idf_target, args.tgt_max_len)
#         idf_info = {
#             "used": True,
#             "idf_csv": str(idf_path),
#             "idf_csv_sha256": sha256_file(idf_path),
#             "policy": "train_fit_only_fixed_before_dev_and_test",
#         }

#     configuration = build_configuration(
#         args=args,
#         checkpoint_info=checkpoint_info,
#         checkpoint_fingerprint=checkpoint_fingerprint,
#         alignment_info=alignment_info,
#         idf_info=idf_info,
#     )
#     validate_test_authorization(args, configuration)

#     print(f"Split: {args.split_name}")
#     print(f"Input CSV: {input_path}")
#     print(f"Device: {device}")
#     print(f"Configuration: {args.configuration_name}")

#     source_embeddings, target_embeddings, cache_used = load_or_encode_embeddings(
#         args=args,
#         input_path=input_path,
#         configuration=configuration,
#         source_texts=source_texts,
#         target_texts=target_texts,
#         source_tokenizer=source_tokenizer,
#         target_tokenizer=target_tokenizer,
#         source_model=source_model,
#         target_model=target_model,
#         source_head=source_head,
#         target_head=target_head,
#         device=device,
#         source_idf=source_idf,
#         target_idf=target_idf,
#         r=r,
#         t=t,
#     )

#     if alignment_info["used"] and alignment_info["stage"] == "sentence":
#         if r is None or t is None:
#             raise RuntimeError("Sentence-level Kabsch requested but R/t are unavailable")
#         source_retrieval_embeddings = apply_kabsch(source_embeddings, r, t)
#     else:
#         source_retrieval_embeddings = source_embeddings

#     print("[3/3] Retrieving candidates...")
#     if args.use_csls:
#         topk_idx, scores = csls_topk(
#             source_retrieval_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#         retrieval = "csls"
#     else:
#         topk_idx, scores = cosine_topk(
#             source_retrieval_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#         )
#         retrieval = "cosine"

#     gold_idx = np.arange(len(dataframe), dtype=np.int64)
#     eval_ks = [int(k) for k in args.eval_ks]
#     metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)
#     predicted_idx = topk_idx[:, 0]

#     output: Dict[str, Any] = {
#         "Bahnaric": raw_source,
#         "Bahnaric_normalized": source_texts,
#         "Predicted_VN": [raw_target[index] for index in predicted_idx],
#         "Gold_VN": raw_target,
#         "Gold_VN_normalized": target_texts,
#         "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
#         "TopK_Preds": [
#             "|".join(raw_target[index] for index in row) for row in topk_idx
#         ],
#         "P@1": (predicted_idx == gold_idx).astype(np.float32).tolist(),
#         "MRR": np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
#         .astype(np.float32)
#         .tolist(),
#         "Top1_score": [
#             float(scores[row, predicted_idx[row]]) for row in range(len(dataframe))
#         ],
#         "Gold_score": [float(scores[row, row]) for row in range(len(dataframe))],
#     }
#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         output[f"Hit@{k_eff}"] = (ranks <= k_eff).astype(np.float32).tolist()

#     predictions = make_bucket_columns(pd.DataFrame(output))
#     predictions.to_csv(output_dir / "sentence_predictions.csv", index=False)
#     save_bucket_metrics(predictions, output_dir, eval_ks)

#     rounded_metrics = {
#         key: round(float(value), 4) for key, value in metrics.items()
#     }
#     rounded_metrics.update(
#         {
#             "schema_version": 1,
#             "family": FAMILY,
#             "split": args.split_name,
#             "configuration_name": args.configuration_name,
#             "configuration": configuration,
#             "evaluator_script": EVALUATOR_SCRIPT,
#             "evaluation_cli_args": build_evaluation_cli_args(args),
#             "method": "xlmr_lora_projection_optional_kabsch",
#             "input_csv": str(input_path),
#             "input_sha256": sha256_file(input_path),
#             "num_queries": int(len(dataframe)),
#             "candidate_pool_size": int(len(dataframe)),
#             "embedding_dim": int(source_retrieval_embeddings.shape[1]),
#             "retrieval": retrieval,
#             "device": str(device),
#             "embedding_cache": cache_used,
#             "selection_policy": "dev_only_then_single_test_evaluation",
#             "training_policy": checkpoint_info.get("training_policy"),
#         }
#     )
#     metrics_path = output_dir / "metrics.json"
#     metrics_path.write_text(
#         json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
#         encoding="utf-8",
#     )
#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
#     print(f"Saved metrics to {metrics_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=(
#             "Development-selected XLM-R LoRA projection/Kabsch retrieval evaluator"
#         )
#     )
#     input_group = parser.add_mutually_exclusive_group(required=True)
#     input_group.add_argument("--input_csv", help="Development or test CSV")
#     input_group.add_argument(
#         "--test_csv",
#         dest="input_csv",
#         help="Deprecated alias for --input_csv",
#     )
#     parser.add_argument("--split_name", choices=["dev", "test"], required=True)
#     parser.add_argument("--configuration_name", required=True)
#     parser.add_argument("--selection_manifest", default=None)
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--proj_dir", required=True)
#     parser.add_argument("--checkpoint_manifest", default=None)
#     parser.add_argument("--allow_unverified_checkpoint", action="store_true")
#     parser.add_argument("--alignment_dir", default=None)
#     parser.add_argument("--alignment_manifest", default=None)
#     parser.add_argument("--require_alignment_manifest", action="store_true")
#     parser.add_argument("--src_model", default="xlm-roberta-base")
#     parser.add_argument("--tgt_model", default="xlm-roberta-base")
#     parser.add_argument(
#         "--pooling",
#         choices=["sentence_mean", "token_mean", "token_idf"],
#         default="token_mean",
#     )
#     parser.add_argument("--src_max_len", type=int, default=256)
#     parser.add_argument("--tgt_max_len", type=int, default=256)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
#     parser.add_argument("--no_kabsch", action="store_true")
#     parser.add_argument(
#         "--kabsch_stage",
#         choices=["sentence", "token"],
#         default="sentence",
#     )
#     parser.add_argument("--use_csls", action="store_true")
#     parser.add_argument("--csls_k", type=int, default=10)
#     parser.add_argument("--use_lora", action="store_true")
#     parser.add_argument("--lowercase", action="store_true")
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")
#     parser.add_argument(
#         "--idf_csv",
#         default=None,
#         help="Training CSV used to build fixed IDF statistics for token_idf pooling",
#     )
#     parser.add_argument("--embedding_cache_dir", default=None)
#     parser.add_argument("--no_cuda", action="store_true")
#     parser.add_argument("--no_mps", action="store_true")
#     main(parser.parse_args())

#####################################
# #!/usr/bin/env python3
# """Reviewer-compliant XLM-R LoRA projection retrieval evaluator.

# Every XLM-R LoRA/projection/Kabsch/retrieval variant must first be evaluated on
# ``data/dev.csv``. Only the exact configuration recorded in
# ``results/dev_selection/selected_configs.json`` may be evaluated on the held-out
# ``data/test.csv``.

# This file evaluates an existing LoRA + projection checkpoint. Training is
# handled by ``src/lora_projection_generalization_train_eval.py``. The evaluator
# supports:

# * sentence-mean, token-mean, and token-IDF pooling;
# * no Kabsch, sentence-level Kabsch, and token-level Kabsch;
# * cosine and CSLS retrieval;
# * resumable embedding caching; and
# * exact test-time authorization against the development-selection manifest.
# """

# from __future__ import annotations

# import argparse
# import hashlib
# import json
# import math
# import os
# import re
# import unicodedata
# from collections import Counter
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer

# try:
#     from peft import PeftModel
# except Exception:
#     PeftModel = None

# FAMILY = "xlmr_lora_projection"
# EVALUATOR_SCRIPT = "src/previous_pipeline_baseline.py"
# REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


# class ProjectionHead(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# def normalize_text(
#     text: str,
#     lowercase: bool = False,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
# ) -> str:
#     text = unicodedata.normalize("NFC", str(text))
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

#     return re.sub(r"\s+", " ", text).strip()


# def read_parallel_csv(path: Path) -> pd.DataFrame:
#     if not path.is_file():
#         raise FileNotFoundError(f"CSV not found: {path}")
#     df = pd.read_csv(path)
#     missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
#     if missing:
#         raise ValueError(f"{path} is missing required columns: {missing}")
#     return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


# def sha256_file(path: Path) -> str:
#     digest = hashlib.sha256()
#     with path.open("rb") as handle:
#         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
#             digest.update(chunk)
#     return digest.hexdigest()


# def sha256_optional_file(path: Optional[Path]) -> Optional[str]:
#     return sha256_file(path) if path is not None and path.is_file() else None


# def canonical_json(value: Any) -> str:
#     return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# def directory_artifact_fingerprint(paths: List[Path]) -> str:
#     records: List[Dict[str, Any]] = []
#     for path in paths:
#         if path.is_file():
#             records.append(
#                 {
#                     "path": str(path),
#                     "size": int(path.stat().st_size),
#                     "sha256": sha256_file(path),
#                 }
#             )
#         elif path.is_dir():
#             for child in sorted(p for p in path.rglob("*") if p.is_file()):
#                 records.append(
#                     {
#                         "path": str(child),
#                         "size": int(child.stat().st_size),
#                         "sha256": sha256_file(child),
#                     }
#                 )
#         else:
#             records.append({"path": str(path), "missing": True})
#     return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


# def get_hidden_size(model: nn.Module) -> int:
#     hidden = getattr(model.config, "hidden_size", None)
#     if hidden is None:
#         hidden = getattr(model.config, "dim", None)
#     if hidden is None:
#         raise ValueError("Could not infer hidden size from model config")
#     return int(hidden)


# def load_projection_head(
#     path: Path,
#     hidden_size: int,
#     device: torch.device,
# ) -> Tuple[nn.Module, int]:
#     if not path.is_file():
#         raise FileNotFoundError(f"Projection head not found: {path}")

#     state_dict = torch.load(path, map_location=device)
#     if "net.0.weight" not in state_dict:
#         raise ValueError(
#             f"{path} is not a ProjectionHead state_dict: missing net.0.weight"
#         )

#     out_dim = int(state_dict["net.0.weight"].shape[0])
#     in_dim = int(state_dict["net.0.weight"].shape[1])
#     if in_dim != hidden_size:
#         raise ValueError(
#             f"Projection input dimension mismatch for {path}: "
#             f"checkpoint={in_dim}, encoder={hidden_size}"
#         )

#     head = ProjectionHead(hidden_size, out_dim).to(device)
#     head.load_state_dict(state_dict, strict=True)
#     head.eval()
#     return head, out_dim


# def maybe_load_lora(base_model: nn.Module, adapters_dir: Path, required: bool) -> nn.Module:
#     if not required:
#         return base_model
#     if not adapters_dir.is_dir():
#         raise FileNotFoundError(f"LoRA adapter directory not found: {adapters_dir}")
#     if not (adapters_dir / "adapter_config.json").is_file():
#         raise FileNotFoundError(
#             f"LoRA adapter_config.json not found in {adapters_dir}"
#         )
#     if PeftModel is None:
#         raise ImportError("peft is required because --use_lora was requested")
#     print(f"Loading LoRA adapter from {adapters_dir}")
#     model = PeftModel.from_pretrained(base_model, str(adapters_dir))
#     model.eval()
#     return model


# def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return x / norms


# def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
#     k = int(max(1, min(topk, scores.shape[1])))
#     if k == scores.shape[1]:
#         return np.argsort(-scores, axis=1)
#     unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
#     selected = np.take_along_axis(scores, unsorted, axis=1)
#     order = np.argsort(-selected, axis=1)
#     return np.take_along_axis(unsorted, order, axis=1)


# def cosine_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     scores = l2_normalize_rows(query) @ l2_normalize_rows(index).T
#     return topk_from_scores(scores, topk), scores


# def csls_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
#     csls_k: int = 10,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)
#     cosine = q @ z.T

#     if q.shape[0] <= 1 or z.shape[0] <= 1:
#         return topk_from_scores(cosine, topk), cosine

#     k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))
#     rq = np.partition(cosine, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
#     rz = np.partition(cosine, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)
#     scores = 2.0 * cosine - rq[:, None] - rz[None, :]
#     return topk_from_scores(scores, topk), scores


# def ranking_metrics(
#     topk_idx: np.ndarray,
#     gold_idx: np.ndarray,
#     eval_ks: List[int],
# ) -> Tuple[Dict[str, float], np.ndarray]:
#     ranks = np.full(topk_idx.shape[0], np.inf, dtype=np.float64)
#     for row in range(topk_idx.shape[0]):
#         hits = np.where(topk_idx[row] == gold_idx[row])[0]
#         if len(hits):
#             ranks[row] = float(hits[0] + 1)

#     metrics: Dict[str, float] = {
#         "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
#         "Top1_acc": float(np.mean(ranks == 1.0)),
#     }
#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         hit = float(np.mean(ranks <= k_eff))
#         metrics[f"Hit@{k_eff}"] = hit
#         metrics[f"Recall@{k_eff}"] = hit
#         metrics[f"Precision@{k_eff}"] = hit / float(k_eff)
#     return metrics, ranks


# def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
#     result = df_out.copy()
#     result["Bahnaric_len_chars"] = result["Bahnaric"].astype(str).str.len()
#     result["Vietnamese_len_chars"] = result["Gold_VN"].astype(str).str.len()
#     result["Vietnamese_len_words"] = (
#         result["Gold_VN"].astype(str).str.split().map(len)
#     )

#     def vn_len_bin(n: int) -> str:
#         if n <= 5:
#             return "1-5"
#         if n <= 15:
#             return "6-15"
#         if n <= 30:
#             return "16-30"
#         return ">30"

#     result["VN_len_bin"] = result["Vietnamese_len_words"].map(vn_len_bin)
#     return result


# def save_bucket_metrics(
#     df_out: pd.DataFrame,
#     output_dir: Path,
#     eval_ks: List[int],
# ) -> None:
#     metric_cols = ["P@1", "MRR"] + [
#         f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns
#     ]
#     available = [column for column in metric_cols if column in df_out.columns]
#     grouped = df_out.groupby("VN_len_bin", dropna=False)
#     bucket_df = grouped[available].mean().reset_index()
#     bucket_df["count"] = grouped.size().values
#     for column in available:
#         bucket_df[column] = bucket_df[column].astype(float).round(4)
#     bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


# def build_idf(tokenizer: Any, texts: List[str], max_len: int) -> Dict[int, float]:
#     document_frequency: Counter[int] = Counter()
#     n_docs = 0
#     for start in range(0, len(texts), 256):
#         encoded = tokenizer(
#             texts[start : start + 256],
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         attention_mask = encoded["attention_mask"]
#         for token_ids, mask in zip(encoded["input_ids"], attention_mask):
#             active_ids = token_ids[mask.bool()].tolist()
#             document_frequency.update(set(int(token_id) for token_id in active_ids))
#             n_docs += 1
#     return {
#         token_id: math.log((n_docs + 1) / (frequency + 1)) + 1.0
#         for token_id, frequency in document_frequency.items()
#     }


# @torch.inference_mode()
# def encode_project_pool(
#     texts: List[str],
#     tokenizer: Any,
#     base_model: nn.Module,
#     projection_head: nn.Module,
#     device: torch.device,
#     max_len: int,
#     batch_size: int,
#     pooling: str,
#     idf_weights: Optional[Dict[int, float]] = None,
#     token_kabsch_r: Optional[np.ndarray] = None,
#     token_kabsch_t: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     base_model.to(device).eval()
#     projection_head.to(device).eval()
#     all_embeddings: List[np.ndarray] = []

#     for start in range(0, len(texts), batch_size):
#         batch = texts[start : start + batch_size]
#         inputs = tokenizer(
#             batch,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = base_model(**inputs, return_dict=True)
#         hidden = outputs.last_hidden_state
#         attention = inputs["attention_mask"].float()

#         if pooling == "sentence_mean":
#             mask = attention.unsqueeze(-1)
#             pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#             embedding = projection_head(pooled)
#         elif pooling in {"token_mean", "token_idf"}:
#             batch_size_actual, sequence_length, hidden_size = hidden.shape
#             projected = projection_head(
#                 hidden.reshape(batch_size_actual * sequence_length, hidden_size)
#             ).reshape(batch_size_actual, sequence_length, -1)

#             if token_kabsch_r is not None or token_kabsch_t is not None:
#                 if token_kabsch_r is None or token_kabsch_t is None:
#                     raise ValueError(
#                         "token_kabsch_r and token_kabsch_t must be provided together"
#                     )
#                 r_tensor = torch.as_tensor(
#                     token_kabsch_r,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )
#                 t_tensor = torch.as_tensor(
#                     token_kabsch_t,
#                     dtype=projected.dtype,
#                     device=projected.device,
#                 )
#                 projected = torch.matmul(projected, r_tensor.T) + t_tensor

#             if pooling == "token_idf":
#                 if idf_weights is None:
#                     raise ValueError("pooling=token_idf requires IDF weights")
#                 token_ids = inputs["input_ids"]
#                 weights = torch.ones_like(token_ids, dtype=torch.float32)
#                 for batch_index in range(batch_size_actual):
#                     for token_index in range(sequence_length):
#                         token_id = int(token_ids[batch_index, token_index].item())
#                         weights[batch_index, token_index] = float(
#                             idf_weights.get(token_id, 1.0)
#                         )
#                 weights = weights.to(device) * attention
#             else:
#                 weights = attention

#             expanded_weights = weights.unsqueeze(-1)
#             embedding = (projected * expanded_weights).sum(dim=1) / (
#                 expanded_weights.sum(dim=1).clamp(min=1e-9)
#             )
#         else:
#             raise ValueError(f"Unknown pooling mode: {pooling}")

#         all_embeddings.append(embedding.detach().cpu().numpy())
#         print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

#     return np.vstack(all_embeddings).astype(np.float32)


# def apply_kabsch(x: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
#     return (r @ x.T).T + t


# def load_checkpoint_manifest(args: argparse.Namespace, proj_dir: Path) -> Dict[str, Any]:
#     manifest_path: Optional[Path]
#     if args.checkpoint_manifest:
#         manifest_path = Path(args.checkpoint_manifest)
#     else:
#         candidate = proj_dir / "training_manifest.json"
#         manifest_path = candidate if candidate.is_file() else None

#     if manifest_path is None:
#         if args.allow_unverified_checkpoint:
#             return {
#                 "verified": False,
#                 "warning": "checkpoint training manifest not provided",
#             }
#         raise FileNotFoundError(
#             "Reviewer-compliant evaluation requires a checkpoint training manifest. "
#             "Retrain with src/lora_projection_generalization_train_eval.py or pass "
#             "--allow_unverified_checkpoint only for legacy reproduction, not for the "
#             "reviewer-response result."
#         )

#     if not manifest_path.is_file():
#         raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
#     manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#     if manifest.get("training_policy") != "train_fit_only_no_dev_refit":
#         if not args.allow_unverified_checkpoint:
#             raise ValueError(
#                 f"Checkpoint manifest {manifest_path} does not declare "
#                 "training_policy=train_fit_only_no_dev_refit"
#             )
#     return {
#         "verified": manifest.get("training_policy") == "train_fit_only_no_dev_refit",
#         "manifest_path": str(manifest_path),
#         "manifest_sha256": sha256_file(manifest_path),
#         "training_policy": manifest.get("training_policy"),
#         "train_csv": manifest.get("train_csv"),
#         "train_csv_sha256": manifest.get("train_csv_sha256"),
#         "num_train_pairs": manifest.get("num_train_pairs"),
#         "train_sample_size": manifest.get("train_sample_size"),
#         "epochs": manifest.get("epochs"),
#         "seed": manifest.get("seed"),
#         "projection_dim": manifest.get("projection_dim"),
#         "pooling": manifest.get("pooling"),
#         "lora_r": manifest.get("lora_r"),
#         "lora_alpha": manifest.get("lora_alpha"),
#         "lora_dropout": manifest.get("lora_dropout"),
#         "learning_rate": manifest.get("learning_rate"),
#         "temperature": manifest.get("temperature"),
#     }


# def load_alignment(
#     args: argparse.Namespace,
#     projection_dim: int,
# ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
#     if args.no_kabsch or args.alignment_dir is None:
#         return None, None, {
#             "used": False,
#             "stage": "none",
#             "alignment_dir": None,
#             "artifact_fingerprint": None,
#         }

#     alignment_dir = Path(args.alignment_dir)
#     r_path = alignment_dir / "R.npy"
#     t_path = alignment_dir / "t.npy"
#     if not r_path.is_file() or not t_path.is_file():
#         raise FileNotFoundError(
#             f"Missing R.npy or t.npy under alignment directory {alignment_dir}"
#         )

#     r = np.load(r_path)
#     t = np.load(t_path)
#     if r.shape != (projection_dim, projection_dim):
#         raise ValueError(
#             f"R shape {r.shape} does not match projection dim {projection_dim}"
#         )
#     if t.shape not in {(projection_dim,), (1, projection_dim)}:
#         raise ValueError(
#             f"t shape {t.shape} does not match projection dim {projection_dim}"
#         )
#     t = t.reshape(-1)

#     if args.kabsch_stage == "token" and args.pooling == "sentence_mean":
#         raise ValueError(
#             "Token-level Kabsch is valid only for token_mean or token_idf pooling"
#         )

#     manifest_path = (
#         Path(args.alignment_manifest)
#         if args.alignment_manifest
#         else alignment_dir / "alignment_manifest.json"
#     )
#     provenance: Dict[str, Any] = {
#         "used": True,
#         "stage": args.kabsch_stage,
#         "alignment_dir": str(alignment_dir),
#         "artifact_fingerprint": directory_artifact_fingerprint([r_path, t_path]),
#         "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
#         "manifest_sha256": sha256_optional_file(manifest_path),
#         "provenance_verified": False,
#     }
#     if manifest_path.is_file():
#         manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#         provenance.update(
#             {
#                 "provenance_verified": bool(
#                     manifest.get("fit_policy")
#                     in {
#                         "train_fit_only",
#                         "independent_training_lexicon_no_dev_or_test",
#                     }
#                 ),
#                 "fit_policy": manifest.get("fit_policy"),
#                 "fit_csv": manifest.get("fit_csv"),
#                 "fit_csv_sha256": manifest.get("fit_csv_sha256"),
#                 "alignment_sample_size": manifest.get("alignment_sample_size"),
#             }
#         )
#     elif args.require_alignment_manifest:
#         raise FileNotFoundError(
#             f"Alignment manifest required but not found: {manifest_path}"
#         )

#     return r.astype(np.float32), t.astype(np.float32), provenance


# def build_configuration(
#     args: argparse.Namespace,
#     checkpoint_info: Dict[str, Any],
#     checkpoint_fingerprint: str,
#     alignment_info: Dict[str, Any],
#     idf_info: Dict[str, Any],
# ) -> Dict[str, Any]:
#     return {
#         "src_model": args.src_model,
#         "tgt_model": args.tgt_model,
#         "proj_dir": args.proj_dir,
#         "checkpoint_fingerprint": checkpoint_fingerprint,
#         "checkpoint_training": checkpoint_info,
#         "pooling": args.pooling,
#         "retrieval": "csls" if args.use_csls else "cosine",
#         "kabsch_used": bool(alignment_info.get("used")),
#         "kabsch_stage": alignment_info.get("stage"),
#         "alignment": alignment_info,
#         "use_lora": bool(args.use_lora),
#         "lowercase": bool(args.lowercase),
#         "strip_accents": bool(args.strip_accents),
#         "remove_punct": bool(args.remove_punct),
#         "idf": idf_info,
#         "src_max_len": int(args.src_max_len),
#         "tgt_max_len": int(args.tgt_max_len),
#         "batch_size": int(args.batch_size),
#         "topk_eval": int(args.topk_eval),
#         "eval_ks": [int(k) for k in args.eval_ks],
#         "use_csls": bool(args.use_csls),
#         "csls_k": int(args.csls_k),
#     }


# def build_evaluation_cli_args(args: argparse.Namespace) -> List[str]:
#     cli = [
#         "--proj_dir",
#         args.proj_dir,
#         "--src_model",
#         args.src_model,
#         "--tgt_model",
#         args.tgt_model,
#         "--pooling",
#         args.pooling,
#         "--batch_size",
#         str(args.batch_size),
#         "--src_max_len",
#         str(args.src_max_len),
#         "--tgt_max_len",
#         str(args.tgt_max_len),
#         "--topk_eval",
#         str(args.topk_eval),
#         "--eval_ks",
#         *[str(k) for k in args.eval_ks],
#         "--csls_k",
#         str(args.csls_k),
#     ]
#     if args.checkpoint_manifest:
#         cli.extend(["--checkpoint_manifest", args.checkpoint_manifest])
#     if args.allow_unverified_checkpoint:
#         cli.append("--allow_unverified_checkpoint")
#     if args.alignment_dir:
#         cli.extend(["--alignment_dir", args.alignment_dir])
#     if args.alignment_manifest:
#         cli.extend(["--alignment_manifest", args.alignment_manifest])
#     if args.require_alignment_manifest:
#         cli.append("--require_alignment_manifest")
#     if args.no_kabsch:
#         cli.append("--no_kabsch")
#     else:
#         cli.extend(["--kabsch_stage", args.kabsch_stage])
#     if args.use_lora:
#         cli.append("--use_lora")
#     if args.use_csls:
#         cli.append("--use_csls")
#     if args.lowercase:
#         cli.append("--lowercase")
#     if args.strip_accents:
#         cli.append("--strip_accents")
#     if args.remove_punct:
#         cli.append("--remove_punct")
#     if args.idf_csv:
#         cli.extend(["--idf_csv", args.idf_csv])
#     if args.embedding_cache_dir:
#         cli.extend(["--embedding_cache_dir", args.embedding_cache_dir])
#     if args.no_cuda:
#         cli.append("--no_cuda")
#     if args.no_mps:
#         cli.append("--no_mps")
#     return cli


# def validate_test_authorization(
#     args: argparse.Namespace,
#     configuration: Dict[str, Any],
# ) -> None:
#     if args.split_name != "test":
#         return
#     if not args.selection_manifest:
#         raise ValueError(
#             "Test evaluation requires --selection_manifest. Select on dev first."
#         )
#     manifest_path = Path(args.selection_manifest)
#     if not manifest_path.is_file():
#         raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
#     manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#     if FAMILY not in manifest:
#         raise ValueError(f"Selection manifest has no {FAMILY!r} entry")
#     selected = manifest[FAMILY]
#     expected_name = selected.get("configuration_name")
#     expected_configuration = selected.get("configuration")
#     if args.configuration_name != expected_name:
#         raise ValueError(
#             f"Unauthorized test configuration {args.configuration_name!r}; "
#             f"development selected {expected_name!r}"
#         )
#     if canonical_json(configuration) != canonical_json(expected_configuration):
#         raise ValueError(
#             "Test arguments or model artifacts do not exactly match the "
#             "development-selected XLM-R LoRA configuration"
#         )


# def cache_key(
#     input_path: Path,
#     configuration: Dict[str, Any],
# ) -> str:
#     cache_configuration = dict(configuration)
#     # Retrieval and sentence-level Kabsch do not change the pre-retrieval base
#     # embeddings, so exclude them to maximize reuse. Token-level Kabsch stays in
#     # the key because it changes source embeddings before pooling.
#     cache_configuration.pop("retrieval", None)
#     cache_configuration.pop("use_csls", None)
#     cache_configuration.pop("csls_k", None)
#     alignment = dict(cache_configuration.get("alignment", {}))
#     if cache_configuration.get("kabsch_stage") == "sentence":
#         alignment = {
#             "used": False,
#             "stage": "none_for_base_embedding_cache",
#         }
#         cache_configuration["kabsch_used"] = False
#         cache_configuration["kabsch_stage"] = "none_for_base_embedding_cache"
#     cache_configuration["alignment"] = alignment
#     payload = {
#         "input_sha256": sha256_file(input_path),
#         "configuration": cache_configuration,
#     }
#     return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:28]


# def load_or_encode_embeddings(
#     args: argparse.Namespace,
#     input_path: Path,
#     configuration: Dict[str, Any],
#     source_texts: List[str],
#     target_texts: List[str],
#     source_tokenizer: Any,
#     target_tokenizer: Any,
#     source_model: nn.Module,
#     target_model: nn.Module,
#     source_head: nn.Module,
#     target_head: nn.Module,
#     device: torch.device,
#     source_idf: Optional[Dict[int, float]],
#     target_idf: Optional[Dict[int, float]],
#     r: Optional[np.ndarray],
#     t: Optional[np.ndarray],
# ) -> Tuple[np.ndarray, np.ndarray, str]:
#     cache_path: Optional[Path] = None
#     if args.embedding_cache_dir:
#         cache_dir = Path(args.embedding_cache_dir)
#         cache_dir.mkdir(parents=True, exist_ok=True)
#         cache_path = cache_dir / f"{cache_key(input_path, configuration)}.npz"
#         if cache_path.is_file():
#             try:
#                 with np.load(cache_path) as data:
#                     source_embeddings = np.asarray(data["source"], dtype=np.float32)
#                     target_embeddings = np.asarray(data["target"], dtype=np.float32)
#                 if (
#                     source_embeddings.shape[0] == len(source_texts)
#                     and target_embeddings.shape[0] == len(target_texts)
#                 ):
#                     print(f"Loaded embedding cache: {cache_path}")
#                     return source_embeddings, target_embeddings, str(cache_path)
#             except (OSError, ValueError, KeyError) as exc:
#                 print(f"Ignoring invalid cache {cache_path}: {exc}")

#     token_level_kabsch = (
#         configuration["kabsch_used"] and configuration["kabsch_stage"] == "token"
#     )
#     print("[1/3] Encoding Bahnaric queries...")
#     source_embeddings = encode_project_pool(
#         texts=source_texts,
#         tokenizer=source_tokenizer,
#         base_model=source_model,
#         projection_head=source_head,
#         device=device,
#         max_len=args.src_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=source_idf,
#         token_kabsch_r=r if token_level_kabsch else None,
#         token_kabsch_t=t if token_level_kabsch else None,
#     )
#     print("[2/3] Encoding Vietnamese candidates...")
#     target_embeddings = encode_project_pool(
#         texts=target_texts,
#         tokenizer=target_tokenizer,
#         base_model=target_model,
#         projection_head=target_head,
#         device=device,
#         max_len=args.tgt_max_len,
#         batch_size=args.batch_size,
#         pooling=args.pooling,
#         idf_weights=target_idf,
#     )

#     if cache_path is not None:
#         temporary = cache_path.with_suffix(".tmp.npz")
#         np.savez_compressed(
#             temporary,
#             source=source_embeddings,
#             target=target_embeddings,
#         )
#         os.replace(temporary, cache_path)
#         print(f"Saved embedding cache: {cache_path}")

#     return (
#         source_embeddings,
#         target_embeddings,
#         str(cache_path) if cache_path is not None else "",
#     )


# def main(args: argparse.Namespace) -> None:
#     input_path = Path(args.input_csv)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if args.split_name == "test" and "test" not in input_path.name.lower():
#         print(
#             "WARNING: split_name=test but input filename does not contain 'test'. "
#             "Authorization still relies on the selection manifest."
#         )

#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.no_mps:
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     proj_dir = Path(args.proj_dir)
#     required_projection_paths = [
#         proj_dir / "src_proj.pt",
#         proj_dir / "tgt_proj.pt",
#     ]
#     if args.use_lora:
#         required_projection_paths.extend(
#             [proj_dir / "src_adapters", proj_dir / "tgt_adapters"]
#         )
#     checkpoint_fingerprint = directory_artifact_fingerprint(required_projection_paths)
#     checkpoint_info = load_checkpoint_manifest(args, proj_dir)

#     dataframe = read_parallel_csv(input_path)
#     raw_source = dataframe["Bahnaric"].astype(str).tolist()
#     raw_target = dataframe["Vietnamese"].astype(str).tolist()
#     source_texts = [
#         normalize_text(
#             text,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_source
#     ]
#     target_texts = [
#         normalize_text(
#             text,
#             lowercase=args.lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_target
#     ]

#     source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
#     target_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
#     source_model = AutoModel.from_pretrained(args.src_model)
#     target_model = AutoModel.from_pretrained(args.tgt_model)
#     source_model = maybe_load_lora(
#         source_model,
#         proj_dir / "src_adapters",
#         required=args.use_lora,
#     )
#     target_model = maybe_load_lora(
#         target_model,
#         proj_dir / "tgt_adapters",
#         required=args.use_lora,
#     )

#     source_head, source_projection_dim = load_projection_head(
#         proj_dir / "src_proj.pt", get_hidden_size(source_model), device
#     )
#     target_head, target_projection_dim = load_projection_head(
#         proj_dir / "tgt_proj.pt", get_hidden_size(target_model), device
#     )
#     if source_projection_dim != target_projection_dim:
#         raise ValueError(
#             f"Projection dimensions differ: {source_projection_dim} vs "
#             f"{target_projection_dim}"
#         )

#     r, t, alignment_info = load_alignment(args, source_projection_dim)

#     source_idf: Optional[Dict[int, float]] = None
#     target_idf: Optional[Dict[int, float]] = None
#     idf_info: Dict[str, Any] = {
#         "used": args.pooling == "token_idf",
#         "idf_csv": None,
#         "idf_csv_sha256": None,
#         "policy": "not_used",
#     }
#     if args.pooling == "token_idf":
#         if not args.idf_csv:
#             raise ValueError(
#                 "pooling=token_idf requires --idf_csv. Use data/train_fit.csv so "
#                 "IDF statistics are fixed before development and test evaluation."
#             )
#         idf_path = Path(args.idf_csv)
#         idf_df = read_parallel_csv(idf_path)
#         idf_source = [
#             normalize_text(
#                 text,
#                 lowercase=args.lowercase,
#                 strip_accents=args.strip_accents,
#                 remove_punct=args.remove_punct,
#             )
#             for text in idf_df["Bahnaric"].astype(str).tolist()
#         ]
#         idf_target = [
#             normalize_text(
#                 text,
#                 lowercase=args.lowercase,
#                 strip_accents=args.strip_accents,
#                 remove_punct=args.remove_punct,
#             )
#             for text in idf_df["Vietnamese"].astype(str).tolist()
#         ]
#         print("Building source IDF weights from fixed training data...")
#         source_idf = build_idf(source_tokenizer, idf_source, args.src_max_len)
#         print("Building target IDF weights from fixed training data...")
#         target_idf = build_idf(target_tokenizer, idf_target, args.tgt_max_len)
#         idf_info = {
#             "used": True,
#             "idf_csv": str(idf_path),
#             "idf_csv_sha256": sha256_file(idf_path),
#             "policy": "train_fit_only_fixed_before_dev_and_test",
#         }

#     configuration = build_configuration(
#         args=args,
#         checkpoint_info=checkpoint_info,
#         checkpoint_fingerprint=checkpoint_fingerprint,
#         alignment_info=alignment_info,
#         idf_info=idf_info,
#     )
#     validate_test_authorization(args, configuration)

#     print(f"Split: {args.split_name}")
#     print(f"Input CSV: {input_path}")
#     print(f"Device: {device}")
#     print(f"Configuration: {args.configuration_name}")

#     source_embeddings, target_embeddings, cache_used = load_or_encode_embeddings(
#         args=args,
#         input_path=input_path,
#         configuration=configuration,
#         source_texts=source_texts,
#         target_texts=target_texts,
#         source_tokenizer=source_tokenizer,
#         target_tokenizer=target_tokenizer,
#         source_model=source_model,
#         target_model=target_model,
#         source_head=source_head,
#         target_head=target_head,
#         device=device,
#         source_idf=source_idf,
#         target_idf=target_idf,
#         r=r,
#         t=t,
#     )

#     if alignment_info["used"] and alignment_info["stage"] == "sentence":
#         if r is None or t is None:
#             raise RuntimeError("Sentence-level Kabsch requested but R/t are unavailable")
#         source_retrieval_embeddings = apply_kabsch(source_embeddings, r, t)
#     else:
#         source_retrieval_embeddings = source_embeddings

#     print("[3/3] Retrieving candidates...")
#     if args.use_csls:
#         topk_idx, scores = csls_topk(
#             source_retrieval_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#         retrieval = "csls"
#     else:
#         topk_idx, scores = cosine_topk(
#             source_retrieval_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#         )
#         retrieval = "cosine"

#     gold_idx = np.arange(len(dataframe), dtype=np.int64)
#     eval_ks = [int(k) for k in args.eval_ks]
#     metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)
#     predicted_idx = topk_idx[:, 0]

#     output: Dict[str, Any] = {
#         "Bahnaric": raw_source,
#         "Bahnaric_normalized": source_texts,
#         "Predicted_VN": [raw_target[index] for index in predicted_idx],
#         "Gold_VN": raw_target,
#         "Gold_VN_normalized": target_texts,
#         "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
#         "TopK_Preds": [
#             "|".join(raw_target[index] for index in row) for row in topk_idx
#         ],
#         "TopK_indices": [
#             "|".join(str(int(index)) for index in row) for row in topk_idx
#         ],
#         "TopK_Scores": [
#             "|".join(
#                 f"{float(scores[row_index, candidate_index]):.10g}"
#                 for candidate_index in row
#             )
#             for row_index, row in enumerate(topk_idx)
#         ],
#         "P@1": (predicted_idx == gold_idx).astype(np.float32).tolist(),
#         "MRR": np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
#         .astype(np.float32)
#         .tolist(),
#         "Top1_score": [
#             float(scores[row, predicted_idx[row]]) for row in range(len(dataframe))
#         ],
#         "Gold_score": [float(scores[row, row]) for row in range(len(dataframe))],
#     }
#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         output[f"Hit@{k_eff}"] = (ranks <= k_eff).astype(np.float32).tolist()

#     predictions = make_bucket_columns(pd.DataFrame(output))
#     predictions.to_csv(output_dir / "sentence_predictions.csv", index=False)
#     save_bucket_metrics(predictions, output_dir, eval_ks)

#     rounded_metrics = {
#         key: round(float(value), 4) for key, value in metrics.items()
#     }
#     rounded_metrics.update(
#         {
#             "schema_version": 1,
#             "family": FAMILY,
#             "split": args.split_name,
#             "configuration_name": args.configuration_name,
#             "configuration": configuration,
#             "evaluator_script": EVALUATOR_SCRIPT,
#             "evaluation_cli_args": build_evaluation_cli_args(args),
#             "method": "xlmr_lora_projection_optional_kabsch",
#             "input_csv": str(input_path),
#             "input_sha256": sha256_file(input_path),
#             "num_queries": int(len(dataframe)),
#             "candidate_pool_size": int(len(dataframe)),
#             "embedding_dim": int(source_retrieval_embeddings.shape[1]),
#             "retrieval": retrieval,
#             "device": str(device),
#             "embedding_cache": cache_used,
#             "selection_policy": "dev_only_then_single_test_evaluation",
#             "training_policy": checkpoint_info.get("training_policy"),
#         }
#     )
#     metrics_path = output_dir / "metrics.json"
#     metrics_path.write_text(
#         json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
#         encoding="utf-8",
#     )
#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
#     print(f"Saved metrics to {metrics_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=(
#             "Development-selected XLM-R LoRA projection/Kabsch retrieval evaluator"
#         )
#     )
#     input_group = parser.add_mutually_exclusive_group(required=True)
#     input_group.add_argument("--input_csv", help="Development or test CSV")
#     input_group.add_argument(
#         "--test_csv",
#         dest="input_csv",
#         help="Deprecated alias for --input_csv",
#     )
#     parser.add_argument("--split_name", choices=["dev", "test"], required=True)
#     parser.add_argument("--configuration_name", required=True)
#     parser.add_argument("--selection_manifest", default=None)
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--proj_dir", required=True)
#     parser.add_argument("--checkpoint_manifest", default=None)
#     parser.add_argument("--allow_unverified_checkpoint", action="store_true")
#     parser.add_argument("--alignment_dir", default=None)
#     parser.add_argument("--alignment_manifest", default=None)
#     parser.add_argument("--require_alignment_manifest", action="store_true")
#     parser.add_argument("--src_model", default="xlm-roberta-base")
#     parser.add_argument("--tgt_model", default="xlm-roberta-base")
#     parser.add_argument(
#         "--pooling",
#         choices=["sentence_mean", "token_mean", "token_idf"],
#         default="token_mean",
#     )
#     parser.add_argument("--src_max_len", type=int, default=256)
#     parser.add_argument("--tgt_max_len", type=int, default=256)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
#     parser.add_argument("--no_kabsch", action="store_true")
#     parser.add_argument(
#         "--kabsch_stage",
#         choices=["sentence", "token"],
#         default="sentence",
#     )
#     parser.add_argument("--use_csls", action="store_true")
#     parser.add_argument("--csls_k", type=int, default=10)
#     parser.add_argument("--use_lora", action="store_true")
#     parser.add_argument("--lowercase", action="store_true")
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")
#     parser.add_argument(
#         "--idf_csv",
#         default=None,
#         help="Training CSV used to build fixed IDF statistics for token_idf pooling",
#     )
#     parser.add_argument("--embedding_cache_dir", default=None)
#     parser.add_argument("--no_cuda", action="store_true")
#     parser.add_argument("--no_mps", action="store_true")
#     main(parser.parse_args())

####################################
#!/usr/bin/env python3
"""Reviewer-compliant XLM-R LoRA projection retrieval evaluator.

Every XLM-R LoRA/projection/Kabsch/retrieval variant must first be evaluated on
``data/dev.csv``. Only the exact configuration recorded in
``results/dev_selection/selected_configs.json`` may be evaluated on the held-out
``data/test.csv``.

This file evaluates an existing LoRA + projection checkpoint. Training is
handled by ``src/lora_projection_generalization_train_eval.py``. The evaluator
supports:

* sentence-mean, token-mean, and token-IDF pooling;
* no Kabsch, sentence-level Kabsch, and token-level Kabsch;
* cosine and CSLS retrieval;
* resumable embedding caching; and
* exact test-time authorization against the development-selection manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

FAMILY = "xlmr_lora_projection"
EVALUATOR_SCRIPT = "src/previous_pipeline_baseline.py"
REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_text(
    text: str,
    lowercase: bool = False,
    strip_accents: bool = False,
    remove_punct: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")

    if lowercase:
        text = text.lower()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    return re.sub(r"\s+", " ", text).strip()


def read_parallel_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_optional_file(path: Optional[Path]) -> Optional[str]:
    return sha256_file(path) if path is not None and path.is_file() else None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def directory_artifact_fingerprint(paths: List[Path]) -> str:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            records.append(
                {
                    "path": str(path),
                    "size": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                records.append(
                    {
                        "path": str(child),
                        "size": int(child.stat().st_size),
                        "sha256": sha256_file(child),
                    }
                )
        else:
            records.append({"path": str(path), "missing": True})
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def get_hidden_size(model: nn.Module) -> int:
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config, "dim", None)
    if hidden is None:
        raise ValueError("Could not infer hidden size from model config")
    return int(hidden)


def load_projection_head(
    path: Path,
    hidden_size: int,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Projection head not found: {path}")

    state_dict = torch.load(path, map_location=device)
    if "net.0.weight" not in state_dict:
        raise ValueError(
            f"{path} is not a ProjectionHead state_dict: missing net.0.weight"
        )

    out_dim = int(state_dict["net.0.weight"].shape[0])
    in_dim = int(state_dict["net.0.weight"].shape[1])
    if in_dim != hidden_size:
        raise ValueError(
            f"Projection input dimension mismatch for {path}: "
            f"checkpoint={in_dim}, encoder={hidden_size}"
        )

    head = ProjectionHead(hidden_size, out_dim).to(device)
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    return head, out_dim


def maybe_load_lora(base_model: nn.Module, adapters_dir: Path, required: bool) -> nn.Module:
    if not required:
        return base_model
    if not adapters_dir.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapters_dir}")
    if not (adapters_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter_config.json not found in {adapters_dir}"
        )
    if PeftModel is None:
        raise ImportError("peft is required because --use_lora was requested")
    print(f"Loading LoRA adapter from {adapters_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapters_dir))
    model.eval()
    return model


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
    k = int(max(1, min(topk, scores.shape[1])))
    if k == scores.shape[1]:
        return np.argsort(-scores, axis=1)
    unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(scores, unsorted, axis=1)
    order = np.argsort(-selected, axis=1)
    return np.take_along_axis(unsorted, order, axis=1)


def cosine_topk(
    query: np.ndarray,
    index: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = l2_normalize_rows(query) @ l2_normalize_rows(index).T
    return topk_from_scores(scores, topk), scores


def csls_topk(
    query: np.ndarray,
    index: np.ndarray,
    topk: int,
    csls_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    cosine = q @ z.T

    if q.shape[0] <= 1 or z.shape[0] <= 1:
        return topk_from_scores(cosine, topk), cosine

    k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))
    rq = np.partition(cosine, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
    rz = np.partition(cosine, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)
    scores = 2.0 * cosine - rq[:, None] - rz[None, :]
    return topk_from_scores(scores, topk), scores


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
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        hit = float(np.mean(ranks <= k_eff))
        metrics[f"Hit@{k_eff}"] = hit
        metrics[f"Recall@{k_eff}"] = hit
        metrics[f"Precision@{k_eff}"] = hit / float(k_eff)
    return metrics, ranks


def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
    result = df_out.copy()
    result["Bahnaric_len_chars"] = result["Bahnaric"].astype(str).str.len()
    result["Vietnamese_len_chars"] = result["Gold_VN"].astype(str).str.len()
    result["Vietnamese_len_words"] = (
        result["Gold_VN"].astype(str).str.split().map(len)
    )

    def vn_len_bin(n: int) -> str:
        if n <= 5:
            return "1-5"
        if n <= 15:
            return "6-15"
        if n <= 30:
            return "16-30"
        return ">30"

    result["VN_len_bin"] = result["Vietnamese_len_words"].map(vn_len_bin)
    return result


def save_bucket_metrics(
    df_out: pd.DataFrame,
    output_dir: Path,
    eval_ks: List[int],
) -> None:
    metric_cols = ["P@1", "MRR"] + [
        f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns
    ]
    available = [column for column in metric_cols if column in df_out.columns]
    grouped = df_out.groupby("VN_len_bin", dropna=False)
    bucket_df = grouped[available].mean().reset_index()
    bucket_df["count"] = grouped.size().values
    for column in available:
        bucket_df[column] = bucket_df[column].astype(float).round(4)
    bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


def build_idf(tokenizer: Any, texts: List[str], max_len: int) -> Dict[int, float]:
    document_frequency: Counter[int] = Counter()
    n_docs = 0
    for start in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[start : start + 256],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"]
        for token_ids, mask in zip(encoded["input_ids"], attention_mask):
            active_ids = token_ids[mask.bool()].tolist()
            document_frequency.update(set(int(token_id) for token_id in active_ids))
            n_docs += 1
    return {
        token_id: math.log((n_docs + 1) / (frequency + 1)) + 1.0
        for token_id, frequency in document_frequency.items()
    }


@torch.inference_mode()
def encode_project_pool(
    texts: List[str],
    tokenizer: Any,
    base_model: nn.Module,
    projection_head: nn.Module,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pooling: str,
    idf_weights: Optional[Dict[int, float]] = None,
    token_kabsch_r: Optional[np.ndarray] = None,
    token_kabsch_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_model.to(device).eval()
    projection_head.to(device).eval()
    all_embeddings: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = base_model(**inputs, return_dict=True)
        hidden = outputs.last_hidden_state
        attention = inputs["attention_mask"].float()

        if pooling == "sentence_mean":
            mask = attention.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embedding = projection_head(pooled)
        elif pooling in {"token_mean", "token_idf"}:
            batch_size_actual, sequence_length, hidden_size = hidden.shape
            projected = projection_head(
                hidden.reshape(batch_size_actual * sequence_length, hidden_size)
            ).reshape(batch_size_actual, sequence_length, -1)

            if token_kabsch_r is not None or token_kabsch_t is not None:
                if token_kabsch_r is None or token_kabsch_t is None:
                    raise ValueError(
                        "token_kabsch_r and token_kabsch_t must be provided together"
                    )
                r_tensor = torch.as_tensor(
                    token_kabsch_r,
                    dtype=projected.dtype,
                    device=projected.device,
                )
                t_tensor = torch.as_tensor(
                    token_kabsch_t,
                    dtype=projected.dtype,
                    device=projected.device,
                )
                projected = torch.matmul(projected, r_tensor.T) + t_tensor

            if pooling == "token_idf":
                if idf_weights is None:
                    raise ValueError("pooling=token_idf requires IDF weights")
                token_ids = inputs["input_ids"]
                weights = torch.ones_like(token_ids, dtype=torch.float32)
                for batch_index in range(batch_size_actual):
                    for token_index in range(sequence_length):
                        token_id = int(token_ids[batch_index, token_index].item())
                        weights[batch_index, token_index] = float(
                            idf_weights.get(token_id, 1.0)
                        )
                weights = weights.to(device) * attention
            else:
                weights = attention

            expanded_weights = weights.unsqueeze(-1)
            embedding = (projected * expanded_weights).sum(dim=1) / (
                expanded_weights.sum(dim=1).clamp(min=1e-9)
            )
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        all_embeddings.append(embedding.detach().cpu().numpy())
        print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings).astype(np.float32)


def apply_kabsch(x: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (r @ x.T).T + t


def load_checkpoint_manifest(args: argparse.Namespace, proj_dir: Path) -> Dict[str, Any]:
    manifest_path: Optional[Path]
    if args.checkpoint_manifest:
        manifest_path = Path(args.checkpoint_manifest)
    else:
        candidate = proj_dir / "training_manifest.json"
        manifest_path = candidate if candidate.is_file() else None

    if manifest_path is None:
        if args.allow_unverified_checkpoint:
            return {
                "verified": False,
                "warning": "checkpoint training manifest not provided",
            }
        raise FileNotFoundError(
            "Reviewer-compliant evaluation requires a checkpoint training manifest. "
            "Retrain with src/lora_projection_generalization_train_eval.py or pass "
            "--allow_unverified_checkpoint only for legacy reproduction, not for the "
            "reviewer-response result."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_policy") != "train_fit_only_no_dev_refit":
        if not args.allow_unverified_checkpoint:
            raise ValueError(
                f"Checkpoint manifest {manifest_path} does not declare "
                "training_policy=train_fit_only_no_dev_refit"
            )
    return {
        "verified": manifest.get("training_policy") == "train_fit_only_no_dev_refit",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_policy": manifest.get("training_policy"),
        "train_csv": manifest.get("train_csv"),
        "train_csv_sha256": manifest.get("train_csv_sha256"),
        "num_train_pairs": manifest.get("num_train_pairs"),
        "train_sample_size": manifest.get("train_sample_size"),
        "epochs": manifest.get("epochs"),
        "seed": manifest.get("seed"),
        "projection_dim": manifest.get("projection_dim"),
        "pooling": manifest.get("pooling"),
        "lora_r": manifest.get("lora_r"),
        "lora_alpha": manifest.get("lora_alpha"),
        "lora_dropout": manifest.get("lora_dropout"),
        "learning_rate": manifest.get("learning_rate"),
        "temperature": manifest.get("temperature"),
    }


def load_alignment(
    args: argparse.Namespace,
    projection_dim: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if args.no_kabsch or args.alignment_dir is None:
        return None, None, {
            "used": False,
            "stage": "none",
            "alignment_dir": None,
            "artifact_fingerprint": None,
        }

    alignment_dir = Path(args.alignment_dir)
    r_path = alignment_dir / "R.npy"
    t_path = alignment_dir / "t.npy"
    if not r_path.is_file() or not t_path.is_file():
        raise FileNotFoundError(
            f"Missing R.npy or t.npy under alignment directory {alignment_dir}"
        )

    r = np.load(r_path)
    t = np.load(t_path)
    if r.shape != (projection_dim, projection_dim):
        raise ValueError(
            f"R shape {r.shape} does not match projection dim {projection_dim}"
        )
    if t.shape not in {(projection_dim,), (1, projection_dim)}:
        raise ValueError(
            f"t shape {t.shape} does not match projection dim {projection_dim}"
        )
    t = t.reshape(-1)

    if args.kabsch_stage == "token" and args.pooling == "sentence_mean":
        raise ValueError(
            "Token-level Kabsch is valid only for token_mean or token_idf pooling"
        )

    manifest_path = (
        Path(args.alignment_manifest)
        if args.alignment_manifest
        else alignment_dir / "alignment_manifest.json"
    )
    provenance: Dict[str, Any] = {
        "used": True,
        "stage": args.kabsch_stage,
        "alignment_dir": str(alignment_dir),
        "artifact_fingerprint": directory_artifact_fingerprint([r_path, t_path]),
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_sha256": sha256_optional_file(manifest_path),
        "provenance_verified": False,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance.update(
            {
                "provenance_verified": bool(
                    manifest.get("fit_policy")
                    in {
                        "train_fit_only",
                        "independent_training_lexicon_no_dev_or_test",
                    }
                ),
                "fit_policy": manifest.get("fit_policy"),
                "fit_csv": manifest.get("fit_csv"),
                "fit_csv_sha256": manifest.get("fit_csv_sha256"),
                "alignment_sample_size": manifest.get("alignment_sample_size"),
            }
        )
    elif args.require_alignment_manifest:
        raise FileNotFoundError(
            f"Alignment manifest required but not found: {manifest_path}"
        )

    return r.astype(np.float32), t.astype(np.float32), provenance


def build_configuration(
    args: argparse.Namespace,
    checkpoint_info: Dict[str, Any],
    checkpoint_fingerprint: str,
    alignment_info: Dict[str, Any],
    idf_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "src_model": args.src_model,
        "tgt_model": args.tgt_model,
        "proj_dir": args.proj_dir,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_training": checkpoint_info,
        "pooling": args.pooling,
        "retrieval": "csls" if args.use_csls else "cosine",
        "kabsch_used": bool(alignment_info.get("used")),
        "kabsch_stage": alignment_info.get("stage"),
        "alignment": alignment_info,
        "use_lora": bool(args.use_lora),
        "lowercase": bool(args.lowercase),
        "strip_accents": bool(args.strip_accents),
        "remove_punct": bool(args.remove_punct),
        "idf": idf_info,
        "src_max_len": int(args.src_max_len),
        "tgt_max_len": int(args.tgt_max_len),
        "batch_size": int(args.batch_size),
        "topk_eval": int(args.topk_eval),
        "eval_ks": [int(k) for k in args.eval_ks],
        "use_csls": bool(args.use_csls),
        "csls_k": int(args.csls_k),
    }


def build_evaluation_cli_args(args: argparse.Namespace) -> List[str]:
    cli = [
        "--proj_dir",
        args.proj_dir,
        "--src_model",
        args.src_model,
        "--tgt_model",
        args.tgt_model,
        "--pooling",
        args.pooling,
        "--batch_size",
        str(args.batch_size),
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
    ]
    if args.checkpoint_manifest:
        cli.extend(["--checkpoint_manifest", args.checkpoint_manifest])
    if args.allow_unverified_checkpoint:
        cli.append("--allow_unverified_checkpoint")
    if args.alignment_dir:
        cli.extend(["--alignment_dir", args.alignment_dir])
    if args.alignment_manifest:
        cli.extend(["--alignment_manifest", args.alignment_manifest])
    if args.require_alignment_manifest:
        cli.append("--require_alignment_manifest")
    if args.no_kabsch:
        cli.append("--no_kabsch")
    else:
        cli.extend(["--kabsch_stage", args.kabsch_stage])
    if args.use_lora:
        cli.append("--use_lora")
    if args.use_csls:
        cli.append("--use_csls")
    if args.lowercase:
        cli.append("--lowercase")
    if args.strip_accents:
        cli.append("--strip_accents")
    if args.remove_punct:
        cli.append("--remove_punct")
    if args.idf_csv:
        cli.extend(["--idf_csv", args.idf_csv])
    if args.embedding_cache_dir:
        cli.extend(["--embedding_cache_dir", args.embedding_cache_dir])
    if args.no_cuda:
        cli.append("--no_cuda")
    if args.no_mps:
        cli.append("--no_mps")
    return cli


def validate_test_authorization(
    args: argparse.Namespace,
    configuration: Dict[str, Any],
) -> None:
    if args.split_name != "test":
        return
    if not args.selection_manifest:
        raise ValueError(
            "Test evaluation requires --selection_manifest. Select on dev first."
        )
    manifest_path = Path(args.selection_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if FAMILY not in manifest:
        raise ValueError(f"Selection manifest has no {FAMILY!r} entry")
    selected = manifest[FAMILY]
    expected_name = selected.get("configuration_name")
    expected_configuration = selected.get("configuration")
    if args.configuration_name != expected_name:
        raise ValueError(
            f"Unauthorized test configuration {args.configuration_name!r}; "
            f"development selected {expected_name!r}"
        )
    if canonical_json(configuration) != canonical_json(expected_configuration):
        raise ValueError(
            "Test arguments or model artifacts do not exactly match the "
            "development-selected XLM-R LoRA configuration"
        )


def cache_key(
    input_path: Path,
    configuration: Dict[str, Any],
) -> str:
    cache_configuration = dict(configuration)
    # Retrieval and sentence-level Kabsch do not change the pre-retrieval base
    # embeddings, so exclude them to maximize reuse. Token-level Kabsch stays in
    # the key because it changes source embeddings before pooling.
    cache_configuration.pop("retrieval", None)
    cache_configuration.pop("use_csls", None)
    cache_configuration.pop("csls_k", None)
    alignment = dict(cache_configuration.get("alignment", {}))
    if cache_configuration.get("kabsch_stage") == "sentence":
        alignment = {
            "used": False,
            "stage": "none_for_base_embedding_cache",
        }
        cache_configuration["kabsch_used"] = False
        cache_configuration["kabsch_stage"] = "none_for_base_embedding_cache"
    cache_configuration["alignment"] = alignment
    payload = {
        "input_sha256": sha256_file(input_path),
        "configuration": cache_configuration,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:28]


def load_or_encode_embeddings(
    args: argparse.Namespace,
    input_path: Path,
    configuration: Dict[str, Any],
    source_texts: List[str],
    target_texts: List[str],
    source_tokenizer: Any,
    target_tokenizer: Any,
    source_model: nn.Module,
    target_model: nn.Module,
    source_head: nn.Module,
    target_head: nn.Module,
    device: torch.device,
    source_idf: Optional[Dict[int, float]],
    target_idf: Optional[Dict[int, float]],
    r: Optional[np.ndarray],
    t: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str]:
    cache_path: Optional[Path] = None
    if args.embedding_cache_dir:
        cache_dir = Path(args.embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key(input_path, configuration)}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path) as data:
                    source_embeddings = np.asarray(data["source"], dtype=np.float32)
                    target_embeddings = np.asarray(data["target"], dtype=np.float32)
                if (
                    source_embeddings.shape[0] == len(source_texts)
                    and target_embeddings.shape[0] == len(target_texts)
                ):
                    print(f"Loaded embedding cache: {cache_path}")
                    return source_embeddings, target_embeddings, str(cache_path)
            except (OSError, ValueError, KeyError) as exc:
                print(f"Ignoring invalid cache {cache_path}: {exc}")

    token_level_kabsch = (
        configuration["kabsch_used"] and configuration["kabsch_stage"] == "token"
    )
    print("[1/3] Encoding Bahnaric queries...")
    source_embeddings = encode_project_pool(
        texts=source_texts,
        tokenizer=source_tokenizer,
        base_model=source_model,
        projection_head=source_head,
        device=device,
        max_len=args.src_max_len,
        batch_size=args.batch_size,
        pooling=args.pooling,
        idf_weights=source_idf,
        token_kabsch_r=r if token_level_kabsch else None,
        token_kabsch_t=t if token_level_kabsch else None,
    )
    print("[2/3] Encoding Vietnamese candidates...")
    target_embeddings = encode_project_pool(
        texts=target_texts,
        tokenizer=target_tokenizer,
        base_model=target_model,
        projection_head=target_head,
        device=device,
        max_len=args.tgt_max_len,
        batch_size=args.batch_size,
        pooling=args.pooling,
        idf_weights=target_idf,
    )

    if cache_path is not None:
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            source=source_embeddings,
            target=target_embeddings,
        )
        os.replace(temporary, cache_path)
        print(f"Saved embedding cache: {cache_path}")

    return (
        source_embeddings,
        target_embeddings,
        str(cache_path) if cache_path is not None else "",
    )


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_name == "test" and "test" not in input_path.name.lower():
        print(
            "WARNING: split_name=test but input filename does not contain 'test'. "
            "Authorization still relies on the selection manifest."
        )

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.no_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    proj_dir = Path(args.proj_dir)
    required_projection_paths = [
        proj_dir / "src_proj.pt",
        proj_dir / "tgt_proj.pt",
    ]
    if args.use_lora:
        required_projection_paths.extend(
            [proj_dir / "src_adapters", proj_dir / "tgt_adapters"]
        )
    checkpoint_fingerprint = directory_artifact_fingerprint(required_projection_paths)
    checkpoint_info = load_checkpoint_manifest(args, proj_dir)

    dataframe = read_parallel_csv(input_path)
    raw_source = dataframe["Bahnaric"].astype(str).tolist()
    raw_target = dataframe["Vietnamese"].astype(str).tolist()
    source_texts = [
        normalize_text(
            text,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_source
    ]
    target_texts = [
        normalize_text(
            text,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_target
    ]

    source_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)
    source_model = AutoModel.from_pretrained(args.src_model)
    target_model = AutoModel.from_pretrained(args.tgt_model)
    source_model = maybe_load_lora(
        source_model,
        proj_dir / "src_adapters",
        required=args.use_lora,
    )
    target_model = maybe_load_lora(
        target_model,
        proj_dir / "tgt_adapters",
        required=args.use_lora,
    )

    source_head, source_projection_dim = load_projection_head(
        proj_dir / "src_proj.pt", get_hidden_size(source_model), device
    )
    target_head, target_projection_dim = load_projection_head(
        proj_dir / "tgt_proj.pt", get_hidden_size(target_model), device
    )
    if source_projection_dim != target_projection_dim:
        raise ValueError(
            f"Projection dimensions differ: {source_projection_dim} vs "
            f"{target_projection_dim}"
        )

    r, t, alignment_info = load_alignment(args, source_projection_dim)

    source_idf: Optional[Dict[int, float]] = None
    target_idf: Optional[Dict[int, float]] = None
    idf_info: Dict[str, Any] = {
        "used": args.pooling == "token_idf",
        "idf_csv": None,
        "idf_csv_sha256": None,
        "policy": "not_used",
    }
    if args.pooling == "token_idf":
        if not args.idf_csv:
            raise ValueError(
                "pooling=token_idf requires --idf_csv. Use data/train_fit.csv so "
                "IDF statistics are fixed before development and test evaluation."
            )
        idf_path = Path(args.idf_csv)
        idf_df = read_parallel_csv(idf_path)
        idf_source = [
            normalize_text(
                text,
                lowercase=args.lowercase,
                strip_accents=args.strip_accents,
                remove_punct=args.remove_punct,
            )
            for text in idf_df["Bahnaric"].astype(str).tolist()
        ]
        idf_target = [
            normalize_text(
                text,
                lowercase=args.lowercase,
                strip_accents=args.strip_accents,
                remove_punct=args.remove_punct,
            )
            for text in idf_df["Vietnamese"].astype(str).tolist()
        ]
        print("Building source IDF weights from fixed training data...")
        source_idf = build_idf(source_tokenizer, idf_source, args.src_max_len)
        print("Building target IDF weights from fixed training data...")
        target_idf = build_idf(target_tokenizer, idf_target, args.tgt_max_len)
        idf_info = {
            "used": True,
            "idf_csv": str(idf_path),
            "idf_csv_sha256": sha256_file(idf_path),
            "policy": "train_fit_only_fixed_before_dev_and_test",
        }

    configuration = build_configuration(
        args=args,
        checkpoint_info=checkpoint_info,
        checkpoint_fingerprint=checkpoint_fingerprint,
        alignment_info=alignment_info,
        idf_info=idf_info,
    )
    validate_test_authorization(args, configuration)

    print(f"Split: {args.split_name}")
    print(f"Input CSV: {input_path}")
    print(f"Device: {device}")
    print(f"Configuration: {args.configuration_name}")

    source_embeddings, target_embeddings, cache_used = load_or_encode_embeddings(
        args=args,
        input_path=input_path,
        configuration=configuration,
        source_texts=source_texts,
        target_texts=target_texts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_model=source_model,
        target_model=target_model,
        source_head=source_head,
        target_head=target_head,
        device=device,
        source_idf=source_idf,
        target_idf=target_idf,
        r=r,
        t=t,
    )

    if alignment_info["used"] and alignment_info["stage"] == "sentence":
        if r is None or t is None:
            raise RuntimeError("Sentence-level Kabsch requested but R/t are unavailable")
        source_retrieval_embeddings = apply_kabsch(source_embeddings, r, t)
    else:
        source_retrieval_embeddings = source_embeddings

    # Save the exact embeddings used by retrieval. The hybrid stage-order
    # experiment needs these to score arbitrary IBM1-first candidates with the
    # already development-selected XLM-R configuration.
    retrieval_embeddings_path = output_dir / "retrieval_embeddings.npz"
    temporary_embeddings_path = output_dir / "retrieval_embeddings.tmp.npz"
    np.savez_compressed(
        temporary_embeddings_path,
        source=np.asarray(source_retrieval_embeddings, dtype=np.float32),
        target=np.asarray(target_embeddings, dtype=np.float32),
    )
    os.replace(temporary_embeddings_path, retrieval_embeddings_path)
    print(f"Saved retrieval embeddings to {retrieval_embeddings_path}")

    print("[3/3] Retrieving candidates...")
    if args.use_csls:
        topk_idx, scores = csls_topk(
            source_retrieval_embeddings,
            target_embeddings,
            topk=args.topk_eval,
            csls_k=args.csls_k,
        )
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(
            source_retrieval_embeddings,
            target_embeddings,
            topk=args.topk_eval,
        )
        retrieval = "cosine"

    gold_idx = np.arange(len(dataframe), dtype=np.int64)
    eval_ks = [int(k) for k in args.eval_ks]
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)
    predicted_idx = topk_idx[:, 0]

    output: Dict[str, Any] = {
        "Bahnaric": raw_source,
        "Bahnaric_normalized": source_texts,
        "Predicted_VN": [raw_target[index] for index in predicted_idx],
        "Gold_VN": raw_target,
        "Gold_VN_normalized": target_texts,
        "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
        "TopK_Preds": [
            "|".join(raw_target[index] for index in row) for row in topk_idx
        ],
        "TopK_indices": [
            "|".join(str(int(index)) for index in row) for row in topk_idx
        ],
        "TopK_Scores": [
            "|".join(
                f"{float(scores[row_index, candidate_index]):.10g}"
                for candidate_index in row
            )
            for row_index, row in enumerate(topk_idx)
        ],
        "P@1": (predicted_idx == gold_idx).astype(np.float32).tolist(),
        "MRR": np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
        .astype(np.float32)
        .tolist(),
        "Top1_score": [
            float(scores[row, predicted_idx[row]]) for row in range(len(dataframe))
        ],
        "Gold_score": [float(scores[row, row]) for row in range(len(dataframe))],
    }
    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        output[f"Hit@{k_eff}"] = (ranks <= k_eff).astype(np.float32).tolist()

    predictions = make_bucket_columns(pd.DataFrame(output))
    predictions.to_csv(output_dir / "sentence_predictions.csv", index=False)
    save_bucket_metrics(predictions, output_dir, eval_ks)

    rounded_metrics = {
        key: round(float(value), 4) for key, value in metrics.items()
    }
    rounded_metrics.update(
        {
            "schema_version": 1,
            "family": FAMILY,
            "split": args.split_name,
            "configuration_name": args.configuration_name,
            "configuration": configuration,
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": build_evaluation_cli_args(args),
            "method": "xlmr_lora_projection_optional_kabsch",
            "input_csv": str(input_path),
            "input_sha256": sha256_file(input_path),
            "num_queries": int(len(dataframe)),
            "candidate_pool_size": int(len(dataframe)),
            "embedding_dim": int(source_retrieval_embeddings.shape[1]),
            "retrieval": retrieval,
            "device": str(device),
            "embedding_cache": cache_used,
            "retrieval_embeddings": str(retrieval_embeddings_path),
            "retrieval_embeddings_sha256": sha256_file(retrieval_embeddings_path),
            "selection_policy": "dev_only_then_single_test_evaluation",
            "training_policy": checkpoint_info.get("training_policy"),
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Development-selected XLM-R LoRA projection/Kabsch retrieval evaluator"
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv", help="Development or test CSV")
    input_group.add_argument(
        "--test_csv",
        dest="input_csv",
        help="Deprecated alias for --input_csv",
    )
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--configuration_name", required=True)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--proj_dir", required=True)
    parser.add_argument("--checkpoint_manifest", default=None)
    parser.add_argument("--allow_unverified_checkpoint", action="store_true")
    parser.add_argument("--alignment_dir", default=None)
    parser.add_argument("--alignment_manifest", default=None)
    parser.add_argument("--require_alignment_manifest", action="store_true")
    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")
    parser.add_argument(
        "--pooling",
        choices=["sentence_mean", "token_mean", "token_idf"],
        default="token_mean",
    )
    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--no_kabsch", action="store_true")
    parser.add_argument(
        "--kabsch_stage",
        choices=["sentence", "token"],
        default="sentence",
    )
    parser.add_argument("--use_csls", action="store_true")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument(
        "--idf_csv",
        default=None,
        help="Training CSV used to build fixed IDF statistics for token_idf pooling",
    )
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_mps", action="store_true")
    main(parser.parse_args())
