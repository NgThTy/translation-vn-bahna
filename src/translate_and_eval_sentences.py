import argparse, logging
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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

def load_head(path, hidden_size, device):
    sd = torch.load(path, map_location=device)
    out_dim, in_dim = sd["net.0.weight"].shape[0], sd["net.0.weight"].shape[1]
    head = ProjectionHead(in_dim=hidden_size, out_dim=out_dim).to(device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head, out_dim

def build_idf(tok, texts, max_len=128):
    """
    Simple IDF over tokenizer input IDs (document frequency by sentence).
    Returns dict: token_id -> idf weight.
    """
    from collections import Counter
    df = Counter()
    n_docs = 0
    for i in range(0, len(texts), 256):
        batch = texts[i:i+256]
        # pad + truncate so we can safely request tensors
        enc = tok(
            batch,
            padding=True,            # <-- changed from False
            truncation=True,
            max_length=max_len,
            return_tensors="pt"      # now it's rectangular, no error
        )
        for ids in enc["input_ids"]:
            n_docs += 1
            df.update(set(ids.tolist()))
    idf = {tid: np.log((n_docs + 1) / (c + 1)) + 1.0 for tid, c in df.items()}
    return idf


@torch.no_grad()
def embed_sentences_token_mapped(
    texts, tok, base, head, device,
    R_np=None, t_np=None, max_len=128, batch=64,
    idf_weights=None  # dict token_id -> idf (optional)
):
    """
    Token -> project -> (optional map) -> IDF-weighted mean pool (or plain mean if idf_weights is None).
    """
    base.to(device).eval(); head.to(device).eval()

    if R_np is not None and t_np is not None:
        R = torch.from_numpy(R_np).to(device).float()                    # (D,D)
        t = torch.from_numpy(t_np).to(device).float().view(1,1,-1)      # (1,1,D)
        do_map = True
    else:
        R, t, do_map = None, None, False

    use_idf = idf_weights is not None
    sent_embs = []

    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        inputs = tok(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        last = base(**inputs, return_dict=True).last_hidden_state   # (B,T,H)
        attn = inputs["attention_mask"].float().unsqueeze(-1)      # (B,T,1)

        B, T, H = last.shape
        tokens_flat = last.reshape(B*T, H)
        tokens_proj = head(tokens_flat).reshape(B, T, -1)           # (B,T,D)

        if do_map:
            tokens_proj = torch.matmul(tokens_proj, R.T) + t        # (B,T,D)

        if use_idf:
            tok_ids = inputs["input_ids"]                           # (B,T)
            w = torch.ones_like(tok_ids, dtype=torch.float32)
            # Fill weights from idf (fallback=1.0)
            for b in range(B):
                for t_i in range(T):
                    tid = int(tok_ids[b, t_i].item())
                    w[b, t_i] = idf_weights.get(tid, 1.0)
            w = w.to(device).unsqueeze(-1)                          # (B,T,1)
            weights = w * attn                                      # zero out padding
        else:
            weights = attn

        D = tokens_proj.size(-1)
        weightsD = weights.expand(-1, -1, D)                        # (B,T,D)
        summed = (tokens_proj * weightsD).sum(dim=1)                # (B,D)
        denom = weights.sum(dim=1).clamp(min=1e-6)                  # (B,1)
        pooled = summed / denom                                     # (B,D)

        sent_embs.append(pooled.detach().cpu().numpy())

    return np.vstack(sent_embs)  # (N,D)

def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1
    return X / n

def cosine_top1(query, index):
    Q = l2norm_rows(query); I = l2norm_rows(index)
    sims = Q @ I.T
    top1 = sims.argmax(axis=1)
    return top1, sims

def csls_top1(src, tgt, k=10):
    """
    CSLS(s,t) = 2*cos(s,t) - r_s - r_t,
    r_s = avg top-k sims of s to tgt; r_t = avg top-k sims of t to src.
    """
    S = l2norm_rows(src)
    T = l2norm_rows(tgt)
    sims = S @ T.T                                 # (Ns, Nt)

    # Safety k
    k = max(1, min(k, T.shape[0]-1, S.shape[0]-1))

    # r_s: mean of top-k over targets for each src row
    rs = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)        # (Ns,)
    # r_t: mean of top-k over sources for each tgt col
    rt = np.partition(sims, -k, axis=0)[-k:, :].mean(axis=0)        # (Nt,)

    csls = 2 * sims - rs[:, None] - rt[None, :]
    top1 = csls.argmax(axis=1)
    return top1, csls

def main(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    LOGGER.info(f"Using device: {device}")

    # Load models + heads
    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
    src_base = AutoModel.from_pretrained(args.src_model)
    tgt_base = AutoModel.from_pretrained(args.tgt_model)
    src_hidden = getattr(src_base.config, "hidden_size", getattr(src_base.config, "dim", None))
    tgt_hidden = getattr(tgt_base.config, "hidden_size", getattr(tgt_base.config, "dim", None))
    src_head, proj_dim = load_head(Path(args.proj_dir) / "src_proj.pt", src_hidden, device)
    tgt_head, _        = load_head(Path(args.proj_dir) / "tgt_proj.pt", tgt_hidden, device)

    # Load R,t
    R = np.load(Path(args.alignment_dir) / "R.npy")
    t = np.load(Path(args.alignment_dir) / "t.npy")
    assert R.shape[0] == R.shape[1] == proj_dim, "R must match projection dim"
    assert t.shape[0] == proj_dim, "t must match projection dim"

    # Read test set (Bahnaric,Vietnamese)
    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric","Vietnamese"])
    bah = df["Bahnaric"].astype(str).tolist()
    vn  = df["Vietnamese"].astype(str).tolist()

    # Optional IDF weights
    src_idf = build_idf(src_tok, bah, max_len=args.src_max_len) if args.use_idf_pool else None
    tgt_idf = build_idf(tgt_tok, vn,  max_len=args.tgt_max_len) if args.use_idf_pool else None

    # --- TOKEN-LEVEL pipeline ---
    LOGGER.info("Embedding Bahnaric (token -> proj -> map -> pool/IDF)...")
    bah_emb = embed_sentences_token_mapped(
        bah, src_tok, src_base, src_head, device,
        R_np=R, t_np=t, max_len=args.src_max_len, batch=args.batch_size,
        idf_weights=src_idf
    )
    LOGGER.info("Embedding Vietnamese (token -> proj -> pool/IDF)...")
    vn_emb  = embed_sentences_token_mapped(
        vn,  tgt_tok, tgt_base, tgt_head, device,
        R_np=None, t_np=None, max_len=args.tgt_max_len, batch=args.batch_size,
        idf_weights=tgt_idf
    )

    # Retrieve
    if args.use_csls:
        LOGGER.info(f"Retrieval: CSLS (k={args.csls_k})")
        pred_idx, _ = csls_top1(bah_emb, vn_emb, k=args.csls_k)
    else:
        LOGGER.info("Retrieval: cosine")
        pred_idx, _ = cosine_top1(bah_emb, vn_emb)

    preds = [vn[i] for i in pred_idx]

    # Eval
    gold = vn
    top1_acc = sum(p == g for p, g in zip(preds, gold)) / max(1, len(gold))
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [gold]).score
        chrf = sacrebleu.corpus_chrf(preds, [gold]).score
    except Exception:
        bleu = None; chrf = None

    print({"Top1_acc": round(top1_acc, 4),
           "BLEU": None if bleu is None else round(bleu, 2),
           "chrF": None if chrf is None else round(chrf, 2)})

    # Save predictions
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Bahnaric": bah, "Predicted_VN": preds, "Gold_VN": gold}).to_csv(out_dir / "sentence_predictions.csv", index=False)
    print(f"Saved predictions to {out_dir/'sentence_predictions.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    ap.add_argument("--proj_dir", required=True, help="dir with src_proj.pt, tgt_proj.pt")
    ap.add_argument("--alignment_dir", required=True, help="dir with R.npy, t.npy")
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
    args = ap.parse_args()
    main(args)
