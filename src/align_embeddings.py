"""
Align Bahnaric -> Vietnamese embeddings with supervised Procrustes (Kabsch)
and evaluate retrieval on a held-out lexicon.

Modes:
1) Default (trained): load and apply ProjectionHeads (src_proj.pt, tgt_proj.pt) from --proj_dir.
2) Baseline (B1): no training, no files:
   --baseline_head {identity,random}
   - identity: projection is identity (requires --proj_dim == hidden_size, e.g., 384 for MiniLM)
   - random: fixed random linear head hidden_size -> proj_dim (frozen)

Use separate lexicons:
- --align_pairs_csv (train) for Kabsch
- --eval_pairs_csv (test) for retrieval evaluation

Outputs:
- Saves R.npy, t.npy to --output_dir
- Logs P@1, P@K, MRR on eval lexicon
- Saves a sample predictions CSV
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from peft import PeftModel

LOGGER = logging.getLogger("align_embeddings")

# -----------------------
# Optional FAISS backend
# -----------------------
try:
    import faiss  # type: ignore
    if platform.system() == "Windows":
        HAS_FAISS = False
        LOGGER.warning(
            "FAISS disabled on Windows to avoid OpenMP runtime conflicts; "
            "falling back to brute-force cosine."
        )
    else:
        HAS_FAISS = True
except Exception:
    HAS_FAISS = False


# -----------------------
# Model bits
# -----------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
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
    elif mode == "random":
        torch.manual_seed(seed)
        head = nn.Linear(hidden_size, out_dim, bias=True)
        for p in head.parameters():
            p.requires_grad = False
        return head
    else:
        raise ValueError(f"Unknown baseline_head: {mode}")


def _infer_proj_dims_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Return (in_dim, out_dim) from a saved ProjectionHead state_dict."""
    w = sd["net.0.weight"]  # [out_dim, in_dim]
    out_dim, in_dim = w.shape[0], w.shape[1]
    return in_dim, out_dim


def load_projection_head(
    path: Path,
    hidden_size: int,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> ProjectionHead:
    sd = torch.load(path, map_location=device)
    in_dim_sd, out_dim = _infer_proj_dims_from_state_dict(sd)
    if in_dim_sd != hidden_size:
        LOGGER.warning(
            f"[{path.name}] state_dict in_dim={in_dim_sd} != base hidden_size={hidden_size}. "
            f"Using hidden_size={hidden_size}; load_state_dict should still work if shapes match."
        )
    head = ProjectionHead(in_dim=hidden_size, out_dim=out_dim, dropout=dropout).to(device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head


# -----------------------
# Embedding utilities
# -----------------------
@torch.no_grad()
def compute_embeddings_from_csv(
    csv_path: str,
    tokenizer,
    base_model: AutoModel,
    proj_head: nn.Module,
    device: torch.device,
    max_len: int = 16,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    """Mean-pool base token embeddings, then pass through projection head."""
    df = pd.read_csv(csv_path)
    if "word" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must have columns: 'word', 'text'")
    words = df["word"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    base_model.to(device).eval()
    proj_head.to(device).eval()

    embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        outs = base_model(**inputs, return_dict=True)
        last = outs.last_hidden_state  # (B, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        summed = (last * mask).sum(dim=1)              # (B, H)
        lens = mask.sum(dim=1).clamp(min=1)            # (B, 1)
        pooled = summed / lens                         # (B, H)
        projected = proj_head(pooled)                  # (B, D)
        embs.append(projected.detach().cpu().numpy())
    embs = np.vstack(embs)
    return words, embs


def kabsch_align(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation R and translation t that minimize ||RX + t - Y||_F^2."""
    assert X.shape == Y.shape, "X and Y must have same shape"
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_Y - R @ mu_X
    return R, t


def map_embeddings(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ X.T).T + t


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def retrieve_topk(embs_query: np.ndarray, embs_index: np.ndarray, topk: int = 5) -> np.ndarray:
    """Cosine similarity (brute-force) top-k."""
    qn = normalize_rows(embs_query)
    inorm = normalize_rows(embs_index)
    sims = qn @ inorm.T
    topk_idx = np.argsort(-sims, axis=1)[:, : topk]
    return topk_idx


def retrieve_topk_faiss(embs_query: np.ndarray, embs_index: np.ndarray, topk: int = 5) -> np.ndarray:
    """Cosine sim via FAISS inner-product on L2-normalized vectors."""
    if not HAS_FAISS:
        raise RuntimeError("FAISS not available")
    tgt = normalize_rows(embs_index).astype("float32")
    q = normalize_rows(embs_query).astype("float32")
    d = tgt.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(tgt)
    index.add(tgt)
    D, I = index.search(q, topk)
    return I


def load_lexicon(lexicon_csv: str) -> Dict[str, set]:
    df = pd.read_csv(lexicon_csv)
    if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
        raise ValueError("Lexicon must have columns: Bahnaric,Vietnamese")
    lex = {}
    for _, row in df.iterrows():
        s = str(row["Bahnaric"])
        t = str(row["Vietnamese"])
        lex.setdefault(s, set()).add(t)
    return lex


def build_pair_matrices(
    align_pairs_csv: str,
    src_words: List[str],
    tgt_words: List[str],
    src_embs: np.ndarray,
    tgt_embs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Select paired rows X (src) and Y (tgt) for alignment from a train lexicon."""
    pairs_df = pd.read_csv(align_pairs_csv).dropna(subset=["Bahnaric", "Vietnamese"])
    src_idx = {w: i for i, w in enumerate(src_words)}
    tgt_idx = {w: i for i, w in enumerate(tgt_words)}
    X_list, Y_list = [], []
    used = 0
    for _, row in pairs_df.iterrows():
        s = str(row["Bahnaric"])
        t = str(row["Vietnamese"])
        if s in src_idx and t in tgt_idx:
            X_list.append(src_embs[src_idx[s]])
            Y_list.append(tgt_embs[tgt_idx[t]])
            used += 1
    if used == 0:
        raise ValueError("No overlapping pairs found between alignment lexicon and vocabularies.")
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y, used


def evaluate_with_lexicon(
    eval_pairs_csv: str,
    src_words: List[str],
    tgt_words: List[str],
    topk_idx: np.ndarray,
) -> Dict[str, float]:
    lex = load_lexicon(eval_pairs_csv)

    topk_k = topk_idx.shape[1] if topk_idx.ndim == 2 else 0
    p1 = pK = 0
    rr_sum = 0.0
    counted = 0

    for i, s in enumerate(src_words):
        correct = lex.get(s)
        if not correct:
            continue
        preds = [tgt_words[j] for j in topk_idx[i]]
        if preds and preds[0] in correct:
            p1 += 1
        if any(p in correct for p in preds[:topk_k]):
            pK += 1
        rank = next((r for r, p in enumerate(preds, 1) if p in correct), None)
        if rank is not None:
            rr_sum += 1.0 / rank
        counted += 1

    denom = max(1, counted)
    return {"P@1": p1 / denom, f"P@{topk_k}": pK / denom, "MRR": rr_sum / denom}


def maybe_load_lora_into_base(base_model, adapters_dir: str):
    """If adapters_dir exists and contains PEFT weights, wrap base_model with PeftModel."""
    if adapters_dir and Path(adapters_dir).is_dir():
        base_model = PeftModel.from_pretrained(base_model, adapters_dir)
        base_model.eval()
    return base_model


def _count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# -----------------------
# Main
# -----------------------
def main(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    LOGGER.info(f"Using device: {device}")

    # In baseline mode, LoRA should be off (no fine-tuning).
    if args.baseline_head != "none" and args.use_lora:
        LOGGER.warning("baseline_head != none: disabling --use_lora (baseline should use pretrained encoder only).")
        args.use_lora = False

    # Load base models + tokenizers
    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
    src_base = AutoModel.from_pretrained(args.src_model)
    tgt_base = AutoModel.from_pretrained(args.tgt_model)

    if args.use_lora:
        if args.proj_dir is None:
            raise ValueError("Need --proj_dir to load LoRA adapters.")
        src_adapters = Path(args.proj_dir) / "src_adapters"
        tgt_adapters = Path(args.proj_dir) / "tgt_adapters"
        src_base = maybe_load_lora_into_base(src_base, str(src_adapters))
        tgt_base = maybe_load_lora_into_base(tgt_base, str(tgt_adapters))

    LOGGER.info(
        f"Trainable params — "
        f"src_base:{_count_trainable(src_base)} "
        f"tgt_base:{_count_trainable(tgt_base)} "
        f"(should be 0 for eval)"
    )

    # Hidden sizes
    src_hidden = getattr(src_base.config, "hidden_size", getattr(src_base.config, "dim", None))
    tgt_hidden = getattr(tgt_base.config, "hidden_size", getattr(tgt_base.config, "dim", None))
    if src_hidden is None or tgt_hidden is None:
        raise ValueError("Cannot determine hidden size from configs.")

    # Projection heads (trained vs baseline)
    if args.baseline_head != "none":
        src_head = make_baseline_head(src_hidden, args.proj_dim, args.baseline_head, seed=args.seed).to(device).eval()
        tgt_head = make_baseline_head(tgt_hidden, args.proj_dim, args.baseline_head, seed=args.seed).to(device).eval()
        LOGGER.info(f"Using baseline head: {args.baseline_head} (proj_dim={args.proj_dim})")
    else:
        if args.proj_dir is None:
            raise ValueError("Need --proj_dir unless --baseline_head is set.")
        proj_dir = Path(args.proj_dir)
        src_proj_path = proj_dir / "src_proj.pt"
        tgt_proj_path = proj_dir / "tgt_proj.pt"
        if not src_proj_path.exists() or not tgt_proj_path.exists():
            raise FileNotFoundError(
                f"Projection heads not found in {proj_dir} (expected src_proj.pt and tgt_proj.pt)."
            )
        src_head = load_projection_head(src_proj_path, hidden_size=src_hidden, dropout=0.1, device=device)
        tgt_head = load_projection_head(tgt_proj_path, hidden_size=tgt_hidden, dropout=0.1, device=device)
        LOGGER.info(f"Loaded projection heads from {proj_dir}")

    # Build embeddings for full vocabularies (projected)
    LOGGER.info("Computing projected embeddings for source/target vocabularies...")
    src_words, src_embs = compute_embeddings_from_csv(
        args.src_emb_csv, src_tok, src_base, src_head, device, max_len=args.src_max_len, batch_size=args.batch_size
    )
    tgt_words, tgt_embs = compute_embeddings_from_csv(
        args.tgt_emb_csv, tgt_tok, tgt_base, tgt_head, device, max_len=args.tgt_max_len, batch_size=args.batch_size
    )
    LOGGER.info(f"Embeddings ready: {len(src_words)} source, {len(tgt_words)} target")

    # Alignment pairs (train) -> compute R, t
    X, Y, used = build_pair_matrices(args.align_pairs_csv, src_words, tgt_words, src_embs, tgt_embs)
    LOGGER.info(f"Using {used} train pairs for Procrustes alignment")
    R, t = kabsch_align(X, Y)

    # Save transformation
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.output_dir) / "R.npy", R)
    np.save(Path(args.output_dir) / "t.npy", t)
    LOGGER.info(f"Saved transformation to {args.output_dir}")

    # Map *all* source embeddings to target space
    src_mapped = map_embeddings(src_embs, R, t)

    # Retrieval over full vocab
    if HAS_FAISS and args.use_faiss:
        LOGGER.info("Retrieval: FAISS (cosine)")
        topk_idx = retrieve_topk_faiss(src_mapped, tgt_embs, topk=args.topk)
    else:
        LOGGER.info("Retrieval: brute-force cosine")
        topk_idx = retrieve_topk(src_mapped, tgt_embs, topk=args.topk)

    # Evaluate on held-out eval lexicon (test)
    eval_stats = evaluate_with_lexicon(args.eval_pairs_csv, src_words, tgt_words, topk_idx)
    LOGGER.info(f"Evaluation (held-out): {eval_stats}")

    # Save a small sample of predictions (first 200)
    out_rows = []
    for i, s in enumerate(src_words[: min(len(src_words), 200)]):
        preds = [tgt_words[j] for j in topk_idx[i]]
        out_rows.append({"Bahnaric": s, "Preds": "|".join(preds)})
    pd.DataFrame(out_rows).to_csv(Path(args.output_dir) / "top_predictions_sample.csv", index=False)
    LOGGER.info(f"Saved sample predictions to {args.output_dir}/top_predictions_sample.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align Bahnaric -> Vietnamese with supervised Procrustes and evaluate on held-out lexicon"
    )
    # Models / tokenizers
    parser.add_argument("--src_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--tgt_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Vocab CSVs
    parser.add_argument("--src_emb_csv", required=True, help="CSV with source vocab: columns word,text")
    parser.add_argument("--tgt_emb_csv", required=True, help="CSV with target vocab: columns word,text")
    # Lexicons (train vs test)
    parser.add_argument("--align_pairs_csv", required=True, help="TRAIN lexicon CSV (Bahnaric,Vietnamese) used to learn R,t")
    parser.add_argument("--eval_pairs_csv", required=True, help="TEST lexicon CSV (Bahnaric,Vietnamese) used ONLY for evaluation")
    # Fine-tuned heads (optional in baseline mode)
    parser.add_argument("--proj_dir", default=None, help="Directory containing src_proj.pt and tgt_proj.pt (optional if using --baseline_head)")
    # Baseline options
    parser.add_argument(
        "--baseline_head",
        choices=["none", "identity", "random"],
        default="none",
        help="If != none, do NOT load src_proj.pt/tgt_proj.pt; use a frozen baseline head instead.",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="Output dim for random baseline head, or for identity must equal hidden size (e.g., 384 for MiniLM).",
    )
    parser.add_argument("--seed", type=int, default=42)
    # I/O and knobs
    parser.add_argument("--output_dir", default="../results/alignment", help="where to save R,t and outputs")
    parser.add_argument("--src_max_len", type=int, default=16)
    parser.add_argument("--tgt_max_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--use_faiss", action="store_true", help="use FAISS for retrieval if installed")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU")
    # LoRA (trained mode only)
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Load LoRA adapters from {proj_dir}/src_adapters and {proj_dir}/tgt_adapters if present (ignored in baseline mode).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    main(args)
