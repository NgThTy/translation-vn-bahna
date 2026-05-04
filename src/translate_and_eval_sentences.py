import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

LOGGER = logging.getLogger("translate_eval")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):  # x: (..., in_dim)
        return self.net(x)


class IdentityHead(nn.Module):
    def forward(self, x):
        return x


def make_baseline_head(hidden_size: int, out_dim: int, mode: str, seed: int = 42) -> nn.Module:
    """
    mode:
      - "identity": only valid if out_dim == hidden_size
      - "random": fixed random linear map hidden_size -> out_dim (frozen)
    """
    if mode == "identity":
        if out_dim != hidden_size:
            raise ValueError(
                f"identity head requires out_dim==hidden_size (got {out_dim} vs {hidden_size})"
            )
        return IdentityHead()
    if mode == "random":
        torch.manual_seed(seed)
        head = nn.Linear(hidden_size, out_dim, bias=True)
        for p in head.parameters():
            p.requires_grad = False
        return head
    raise ValueError(f"Unknown baseline_head: {mode}")


def load_head(path, hidden_size, device):
    sd = torch.load(path, map_location=device)
    out_dim, in_dim = sd["net.0.weight"].shape[0], sd["net.0.weight"].shape[1]
    head = ProjectionHead(in_dim=hidden_size, out_dim=out_dim).to(device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head, out_dim


def maybe_load_lora_into_base(base_model, adapters_dir: str):
    """If adapters_dir exists and contains PEFT weights, wrap base_model with PeftModel."""
    adir = Path(adapters_dir)
    if adir.is_dir():
        base_model = PeftModel.from_pretrained(base_model, str(adir))
        base_model.eval()
    return base_model


def _count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_idf(tok, texts, max_len=128):
    """
    Simple IDF over tokenizer input IDs (document frequency by sentence).
    Returns dict: token_id -> idf weight.
    """
    from collections import Counter

    df = Counter()
    n_docs = 0
    for i in range(0, len(texts), 256):
        batch = texts[i : i + 256]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        for ids in enc["input_ids"]:
            n_docs += 1
            df.update(set(ids.tolist()))
    idf = {tid: np.log((n_docs + 1) / (c + 1)) + 1.0 for tid, c in df.items()}
    return idf


@torch.no_grad()
def embed_sentences_token_mapped(
    texts,
    tok,
    base,
    head,
    device,
    R_np=None,
    t_np=None,
    max_len=128,
    batch=64,
    idf_weights=None,
):
    """
    Token -> project -> (optional map) -> IDF-weighted mean pool (or plain mean if idf_weights is None).
    """
    base.to(device).eval()
    head.to(device).eval()

    if R_np is not None and t_np is not None:
        R = torch.from_numpy(R_np).to(device).float()                 # (D,D)
        t = torch.from_numpy(t_np).to(device).float().view(1, 1, -1)  # (1,1,D)
        do_map = True
    else:
        R, t, do_map = None, None, False

    use_idf = idf_weights is not None
    sent_embs = []

    for i in range(0, len(texts), batch):
        batch_texts = texts[i : i + batch]
        inputs = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        last = base(**inputs, return_dict=True).last_hidden_state  # (B,T,H)
        attn = inputs["attention_mask"].float().unsqueeze(-1)      # (B,T,1)

        B, T, H = last.shape
        tokens_flat = last.reshape(B * T, H)
        tokens_proj = head(tokens_flat).reshape(B, T, -1)          # (B,T,D)

        if do_map:
            tokens_proj = torch.matmul(tokens_proj, R.T) + t       # (B,T,D)

        if use_idf:
            tok_ids = inputs["input_ids"]                          # (B,T)
            w = torch.ones_like(tok_ids, dtype=torch.float32)
            for b in range(B):
                for t_i in range(T):
                    tid = int(tok_ids[b, t_i].item())
                    w[b, t_i] = idf_weights.get(tid, 1.0)
            w = w.to(device).unsqueeze(-1)                         # (B,T,1)
            weights = w * attn
        else:
            weights = attn

        D = tokens_proj.size(-1)
        weightsD = weights.expand(-1, -1, D)                       # (B,T,D)
        summed = (tokens_proj * weightsD).sum(dim=1)               # (B,D)
        denom = weights.sum(dim=1).clamp(min=1e-6)                 # (B,1)
        pooled = summed / denom                                    # (B,D)

        sent_embs.append(pooled.detach().cpu().numpy())

    return np.vstack(sent_embs)  # (N,D)


def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def cosine_topk(query, index, k: int):
    Q = l2norm_rows(query)
    I = l2norm_rows(index)
    sims = Q @ I.T
    k = max(1, min(k, sims.shape[1]))
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    return topk_idx, sims


def csls_topk(src, tgt, k_retrieval: int, k_csls: int = 10):
    """
    CSLS(s,t) = 2*cos(s,t) - r_s - r_t,
    r_s = avg top-k_csls sims of s to tgt; r_t = avg top-k_csls sims of t to src.
    Returns top-k_retrieval indices by CSLS score.
    """
    S = l2norm_rows(src)
    T = l2norm_rows(tgt)
    sims = S @ T.T

    k_csls = max(1, min(k_csls, T.shape[0] - 1, S.shape[0] - 1))
    rs = np.partition(sims, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
    rt = np.partition(sims, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)

    csls = 2 * sims - rs[:, None] - rt[None, :]
    k_retrieval = max(1, min(k_retrieval, csls.shape[1]))
    topk_idx = np.argsort(-csls, axis=1)[:, :k_retrieval]
    return topk_idx, csls


def ranking_metrics(topk_idx: np.ndarray, gold_idx: np.ndarray, ks=(5, 10)):
    """
    topk_idx: (N, Kmax)
    gold_idx: (N,)
    For 1 gold item per query:
      - MRR uses the rank if found, else 0.
      - Recall@K == Hit@K
      - Precision@K == Hit@K / K
    """
    N, Kmax = topk_idx.shape
    ranks = np.full(N, np.inf, dtype=np.float64)

    for i in range(N):
        hits = np.where(topk_idx[i] == gold_idx[i])[0]
        if len(hits) > 0:
            ranks[i] = float(hits[0] + 1)  # 1-based rank

    mrr = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
    out = {"MRR": mrr}

    for K in ks:
        K = int(min(max(1, K), Kmax))
        hitK = float(np.mean(ranks <= K))
        out[f"Hit@{K}"] = hitK
        out[f"Recall@{K}"] = hitK
        out[f"Precision@{K}"] = hitK / float(K)

    return out, ranks


def load_lexicon_pairs(lexicon_csv: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Load a synonym-rich lexicon CSV with columns Bahnaric,Vietnamese and build:
      - src2tgtset: Bahnaric -> set(Vietnamese)
      - tgt2srcset: Vietnamese -> set(Bahnaric)
    """
    df = pd.read_csv(lexicon_csv).dropna(subset=["Bahnaric", "Vietnamese"])
    if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
        raise ValueError("Lexicon must have columns: Bahnaric,Vietnamese")

    src2tgt: Dict[str, Set[str]] = {}
    tgt2src: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        s = str(row["Bahnaric"])
        t = str(row["Vietnamese"])
        src2tgt.setdefault(s, set()).add(t)
        tgt2src.setdefault(t, set()).add(s)

    return src2tgt, tgt2src


def mapping_type_label(
    src: str,
    tgt: str,
    src2tgt: Dict[str, Set[str]],
    tgt2src: Dict[str, Set[str]],
) -> str:
    """
    Determine mapping type bucket for a (src, tgt) pair using lexicon relations.
    Buckets:
      - one-to-one
      - one-to-many
      - many-to-one
      - many-to-many (optional catch-all)
      - unknown (if src or tgt not found in lexicon maps)
    """
    sset = src2tgt.get(src)
    tset = tgt2src.get(tgt)
    if not sset or not tset:
        return "unknown"

    sdeg = len(sset)
    tdeg = len(tset)

    if sdeg == 1 and tdeg == 1:
        return "one-to-one"
    if sdeg > 1 and tdeg == 1:
        return "one-to-many"
    if sdeg == 1 and tdeg > 1:
        return "many-to-one"
    return "many-to-many"


def vn_len_bin_label(vn_text: str) -> Tuple[int, str]:
    """
    Bin Vietnamese gloss length by whitespace tokens:
      - 1
      - 2-3
      - 4-6
      - >6
    Returns (vn_len, bin_label).
    """
    vn_len = len(str(vn_text).split())
    if vn_len <= 1:
        return vn_len, "1"
    if vn_len <= 3:
        return vn_len, "2-3"
    if vn_len <= 6:
        return vn_len, "4-6"
    return vn_len, ">6"


def bucket_summary_table(
    df_out: pd.DataFrame,
    bucket_col: str,
    eval_ks: List[int],
) -> pd.DataFrame:
    """
    Aggregate P@1 / MRR / Hit@K / BERTScore per bucket.

    IMPORTANT FIX:
      - Do NOT aggregate a fake "count" column.
      - Use grouped.size() to compute bucket size safely.
    """
    tmp = df_out.copy()

    # Ensure numeric columns are numeric for groupby mean
    numeric_cols = ["P@1", "MRR"] + [f"Hit@{K}" for K in eval_ks] + ["BERTScore_P", "BERTScore_R", "BERTScore_F1"]
    for col in numeric_cols:
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    grouped = tmp.groupby(bucket_col, dropna=False)

    # Build aggregation dict only with existing columns
    agg: Dict[str, str] = {"P@1": "mean", "MRR": "mean"}
    for K in eval_ks:
        col = f"Hit@{K}"
        if col in tmp.columns:
            agg[col] = "mean"

    # BERTScore means (if present)
    for col in ["BERTScore_P", "BERTScore_R", "BERTScore_F1"]:
        if col in tmp.columns:
            agg[col] = "mean"

    summary = grouped.agg(agg).reset_index()
    summary["count"] = grouped.size().values  # ✅ robust bucket size

    # Round for readability
    for col in summary.columns:
        if col not in [bucket_col, "count"]:
            summary[col] = summary[col].astype(float).round(4)

    return summary.sort_values(by="count", ascending=False)


def main(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    LOGGER.info(f"Using device: {device}")

    # In baseline mode, LoRA should be off (no fine-tuning).
    if args.baseline_head != "none" and args.use_lora:
        LOGGER.warning("baseline_head != none: disabling --use_lora (baseline should use pretrained encoder only).")
        args.use_lora = False

    # Load models
    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
    src_base = AutoModel.from_pretrained(args.src_model)
    tgt_base = AutoModel.from_pretrained(args.tgt_model)

    if args.use_lora:
        if args.proj_dir is None:
            raise ValueError("Need --proj_dir to load LoRA adapters.")
        src_base = maybe_load_lora_into_base(src_base, Path(args.proj_dir) / "src_adapters")
        tgt_base = maybe_load_lora_into_base(tgt_base, Path(args.proj_dir) / "tgt_adapters")

    LOGGER.info(
        f"Trainable params — src_base:{_count_trainable(src_base)} "
        f"tgt_base:{_count_trainable(tgt_base)} (should be 0 for eval)"
    )

    src_hidden = getattr(src_base.config, "hidden_size", getattr(src_base.config, "dim", None))
    tgt_hidden = getattr(tgt_base.config, "hidden_size", getattr(tgt_base.config, "dim", None))
    if src_hidden is None or tgt_hidden is None:
        raise ValueError("Cannot determine hidden size from configs.")

    # Heads (trained vs baseline)
    if args.baseline_head != "none":
        src_head = make_baseline_head(src_hidden, args.proj_dim, args.baseline_head, seed=args.seed).to(device).eval()
        tgt_head = make_baseline_head(tgt_hidden, args.proj_dim, args.baseline_head, seed=args.seed).to(device).eval()
        proj_dim = args.proj_dim
        LOGGER.info(f"Using baseline head: {args.baseline_head} (proj_dim={args.proj_dim})")
    else:
        if args.proj_dir is None:
            raise ValueError("Need --proj_dir unless --baseline_head is set.")
        src_head, proj_dim = load_head(Path(args.proj_dir) / "src_proj.pt", src_hidden, device)
        tgt_head, _ = load_head(Path(args.proj_dir) / "tgt_proj.pt", tgt_hidden, device)

    # -----------------------------
    # (B3) No-Kabsch: optional R,t
    # -----------------------------
    use_kabsch = (not args.no_kabsch) and (args.alignment_dir is not None)

    if use_kabsch:
        R = np.load(Path(args.alignment_dir) / "R.npy")
        t = np.load(Path(args.alignment_dir) / "t.npy")
        assert R.shape[0] == R.shape[1] == proj_dim, "R must match projection dim"
        assert t.shape[0] == proj_dim, "t must match projection dim"
    else:
        R, t = None, None
        LOGGER.info("B3: No-Kabsch mode enabled (no R,t mapping will be applied).")

    # Read test set (Bahnaric,Vietnamese)
    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"])
    bah = df["Bahnaric"].astype(str).tolist()
    vn = df["Vietnamese"].astype(str).tolist()

    # Optional IDF weights
    src_idf = build_idf(src_tok, bah, max_len=args.src_max_len) if args.use_idf_pool else None
    tgt_idf = build_idf(tgt_tok, vn, max_len=args.tgt_max_len) if args.use_idf_pool else None

    # --- TOKEN-LEVEL pipeline ---
    if use_kabsch:
        LOGGER.info("Embedding Bahnaric (token -> proj -> map -> pool/IDF)...")
    else:
        LOGGER.info("Embedding Bahnaric (token -> proj -> pool/IDF) [NO-KABSCH]...")

    bah_emb = embed_sentences_token_mapped(
        bah,
        src_tok,
        src_base,
        src_head,
        device,
        R_np=R,
        t_np=t,
        max_len=args.src_max_len,
        batch=args.batch_size,
        idf_weights=src_idf,
    )

    LOGGER.info("Embedding Vietnamese (token -> proj -> pool/IDF)...")
    vn_emb = embed_sentences_token_mapped(
        vn,
        tgt_tok,
        tgt_base,
        tgt_head,
        device,
        R_np=None,
        t_np=None,
        max_len=args.tgt_max_len,
        batch=args.batch_size,
        idf_weights=tgt_idf,
    )

    # Retrieve (TOP-K)
    k_eval = int(max(1, min(args.topk_eval, len(vn))))
    if args.use_csls:
        LOGGER.info(f"Retrieval: CSLS (csls_k={args.csls_k}) topk_eval={k_eval}")
        topk_idx, _ = csls_topk(bah_emb, vn_emb, k_retrieval=k_eval, k_csls=args.csls_k)
    else:
        LOGGER.info(f"Retrieval: cosine topk_eval={k_eval}")
        topk_idx, _ = cosine_topk(bah_emb, vn_emb, k=k_eval)

    # 1-1 aligned: gold index is row position
    gold_idx = np.arange(len(vn), dtype=np.int64)

    # Top-1 predictions (for BLEU/chrF/BERTScore compatibility)
    pred_top1_idx = topk_idx[:, 0]
    preds = [vn[i] for i in pred_top1_idx]
    gold = vn

    # Rank metrics (global + ranks vector)
    eval_ks = [int(k) for k in args.eval_ks]
    rank_stats, ranks = ranking_metrics(topk_idx, gold_idx, ks=tuple(eval_ks))

    # Per-row retrieval stats (for bucketing)
    is_top1 = (pred_top1_idx == gold_idx).astype(np.float32)  # 0/1
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)  # 0 if not found

    hit_cols: Dict[int, np.ndarray] = {}
    for K in eval_ks:
        K_eff = int(min(max(1, K), topk_idx.shape[1]))
        hit_cols[K] = (ranks <= float(K_eff)).astype(np.float32)

    # Top1 acc (exact match)
    top1_acc = float(np.mean(is_top1)) if len(gold_idx) > 0 else 0.0

    # BLEU & chrF via sacrebleu (Top-1 predictions only)
    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(preds, [gold]).score
        chrf = sacrebleu.corpus_chrf(preds, [gold]).score
    except Exception:
        bleu = None
        chrf = None

    # BERTScore (Top-1 predictions only)
    try:
        from bert_score import score as bert_score

        P, Rm, F1 = bert_score(
            preds,
            gold,
            lang="vi",
            rescale_with_baseline=True,
        )
        bert_p_list = P.tolist()
        bert_r_list = Rm.tolist()
        bert_f1_list = F1.tolist()
        bert_p = float(P.mean())
        bert_r = float(Rm.mean())
        bert_f1 = float(F1.mean())
    except Exception as e:
        LOGGER.warning(f"Could not compute BERTScore: {e}")
        bert_p_list = bert_r_list = bert_f1_list = []
        bert_p = bert_r = bert_f1 = None

    out_metrics = {
        "Top1_acc": round(top1_acc, 4),
        "MRR": round(rank_stats["MRR"], 4),
    }
    for K in eval_ks:
        out_metrics[f"Hit@{K}"] = round(rank_stats.get(f"Hit@{K}", 0.0), 4)
        out_metrics[f"Recall@{K}"] = round(rank_stats.get(f"Recall@{K}", 0.0), 4)
        out_metrics[f"Precision@{K}"] = round(rank_stats.get(f"Precision@{K}", 0.0), 4)

    out_metrics.update(
        {
            "BLEU": None if bleu is None else round(bleu, 2),
            "chrF": None if chrf is None else round(chrf, 2),
            "BERTScore_P": None if bert_p is None else round(bert_p, 4),
            "BERTScore_R": None if bert_r is None else round(bert_r, 4),
            "BERTScore_F1": None if bert_f1 is None else round(bert_f1, 4),
        }
    )

    print(out_metrics)

    # Save predictions + rank + BERTScore
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # store rank of gold in top-k (None if not found)
    gold_rank = [None if not np.isfinite(r) else int(r) for r in ranks.tolist()]

    # store top-k candidate strings (useful for qualitative analysis)
    topk_preds_str = ["|".join([vn[j] for j in row]) for row in topk_idx]

    # Bucket labels (A1/A2)
    mapping_types: Optional[List[str]] = None
    vn_len_list: Optional[List[int]] = None
    vn_len_bin_list: Optional[List[str]] = None

    if args.bucket_eval:
        do_mapping = args.bucket_by in ("mapping_type", "both")
        do_vnlen = args.bucket_by in ("vn_len", "both")

        if do_mapping:
            if not args.lexicon_csv:
                raise ValueError("--bucket_eval with bucket_by mapping_type/both requires --lexicon_csv")
            src2tgt, tgt2src = load_lexicon_pairs(args.lexicon_csv)
            mapping_types = [
                mapping_type_label(bah[i], vn[i], src2tgt, tgt2src) for i in range(len(bah))
            ]

        if do_vnlen:
            vn_len_list = []
            vn_len_bin_list = []
            for text in vn:
                L, b = vn_len_bin_label(text)
                vn_len_list.append(L)
                vn_len_bin_list.append(b)

    # Build output dataframe with per-row metrics (for controlled analysis)
    df_dict = {
        "Bahnaric": bah,
        "Predicted_VN": preds,          # Top-1
        "Gold_VN": gold,
        "Gold_rank": gold_rank,         # 1-based rank within topk_eval (None if not found)
        "TopK_Preds": topk_preds_str,   # pipe-separated top-k predictions
        "P@1": is_top1.tolist(),
        "MRR": rr.tolist(),
    }
    for K in eval_ks:
        df_dict[f"Hit@{K}"] = hit_cols[K].tolist()

    # Add buckets if enabled
    if mapping_types is not None:
        df_dict["Mapping_type"] = mapping_types
    if vn_len_list is not None and vn_len_bin_list is not None:
        df_dict["VN_len"] = vn_len_list
        df_dict["VN_len_bin"] = vn_len_bin_list

    # Add BERTScore columns (Top-1 only)
    df_dict["BERTScore_P"] = bert_p_list if bert_p_list else [None] * len(bah)
    df_dict["BERTScore_R"] = bert_r_list if bert_r_list else [None] * len(bah)
    df_dict["BERTScore_F1"] = bert_f1_list if bert_f1_list else [None] * len(bah)

    df_out = pd.DataFrame(df_dict)
    df_out.to_csv(out_dir / "sentence_predictions.csv", index=False)
    print(f"Saved predictions to {out_dir/'sentence_predictions.csv'}")

    # Bucket summaries (A1/A2)
    if args.bucket_eval:
        summaries = []

        if "Mapping_type" in df_out.columns:
            s_map = bucket_summary_table(df_out, "Mapping_type", eval_ks=eval_ks)
            s_map.to_csv(out_dir / "bucket_metrics_mapping_type.csv", index=False)
            print("\n[Buckets: Mapping_type]")
            print(s_map.to_string(index=False))
            summaries.append(("mapping_type", s_map))

        if "VN_len_bin" in df_out.columns:
            # Keep bins in requested order
            cat_order = ["1", "2-3", "4-6", ">6"]
            df_out["VN_len_bin"] = pd.Categorical(df_out["VN_len_bin"], categories=cat_order, ordered=True)
            s_len = bucket_summary_table(df_out, "VN_len_bin", eval_ks=eval_ks)
            s_len = s_len.sort_values("VN_len_bin")
            s_len.to_csv(out_dir / "bucket_metrics_vn_len.csv", index=False)
            print("\n[Buckets: VN_len_bin]")
            print(s_len.to_string(index=False))
            summaries.append(("vn_len", s_len))

        # Optional combined summary file if both exist
        if len(summaries) > 1:
            combined_rows = []
            for tag, sdf in summaries:
                tmp = sdf.copy()
                tmp.insert(0, "bucket_family", tag)
                combined_rows.append(tmp)
            combined = pd.concat(combined_rows, axis=0, ignore_index=True)
            combined.to_csv(out_dir / "bucket_metrics.csv", index=False)
            print(f"\nSaved combined bucket metrics to {out_dir/'bucket_metrics.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    ap.add_argument(
        "--proj_dir",
        default=None,
        help="dir with src_proj.pt, tgt_proj.pt (optional if using --baseline_head)",
    )
    ap.add_argument("--baseline_head", choices=["none", "identity", "random"], default="none")
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    # (B3) make alignment optional
    ap.add_argument(
        "--alignment_dir",
        default=None,
        help="dir with R.npy, t.npy (omit for B3 no-Kabsch)",
    )
    ap.add_argument(
        "--no_kabsch",
        action="store_true",
        help="Skip loading/applying Kabsch alignment (B3). If set, R,t are ignored even if --alignment_dir is provided.",
    )

    ap.add_argument("--src_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--tgt_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--src_max_len", type=int, default=256)
    ap.add_argument("--tgt_max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output_dir", default="../results/sent_eval")

    ap.add_argument("--use_idf_pool", action="store_true", help="Use IDF-weighted token pooling")
    ap.add_argument("--use_csls", action="store_true", help="Use CSLS retrieval")
    ap.add_argument("--csls_k", type=int, default=10, help="Neighborhood size for CSLS")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument(
        "--use_lora",
        action="store_true",
        help="Load LoRA adapters from {proj_dir}/src_adapters and {proj_dir}/tgt_adapters if present (ignored in baseline mode).",
    )

    # (1.2) rank-based retrieval metrics
    ap.add_argument("--topk_eval", type=int, default=10, help="Compute ranking metrics up to this K")
    ap.add_argument("--eval_ks", type=int, nargs="+", default=[5, 10], help="K values for P@K / R@K")

    # (1.3) bucketed analysis (A1/A2)
    ap.add_argument("--bucket_eval", action="store_true", help="Report metrics per bucket (mapping type / VN length)")
    ap.add_argument(
        "--bucket_by",
        choices=["mapping_type", "vn_len", "both"],
        default="both",
        help="Which bucket family to compute",
    )
    ap.add_argument(
        "--lexicon_csv",
        default=None,
        help=(
            "Lexicon CSV (Bahnaric,Vietnamese) used to infer mapping type buckets. "
            "Example 10K: /Users/tynguyen/Library/CloudStorage/OneDrive-RMITUniversity/RESEARCH/NUS/"
            "multilingual_llm/translation-vn-bahna/data/lexicon.csv"
        ),
    )

    args = ap.parse_args()
    main(args)
