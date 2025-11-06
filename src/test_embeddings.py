import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Reuse dataset, collate, builders, and evaluate from train file
from train_embeddings import (
    BahVnPairsDataset,
    collate_fn,
    build_models,
    evaluate,
)

def maybe_load_lora_into_base(base_model, adapters_dir: str):
    """
    If adapters_dir exists and contains PEFT weights, wrap base_model with PeftModel.
    """
    if adapters_dir and os.path.isdir(adapters_dir):
        base_model = PeftModel.from_pretrained(base_model, adapters_dir)
        base_model.eval()
    return base_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="../data/test.csv")
    ap.add_argument("--output_dir", default="../results/models_demo")
    ap.add_argument("--src_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--tgt_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--projection_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--src_max_len", type=int, default=256)
    ap.add_argument("--tgt_max_len", type=int, default=256)
    ap.add_argument("--loss_type", choices=["mse", "infonce", "hybrid"], default="infonce")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--alpha", type=float, default=0.5)  # for hybrid
    ap.add_argument("--use_lora", action="store_true", help="Force-load LoRA adapters if present")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # --- load dataset + tokenizers ---
    test_ds = BahVnPairsDataset(args.test_csv)
    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_fn(
            b, src_tok, tgt_tok, args.src_max_len, args.tgt_max_len, device
        ),
    )

    # --- rebuild model skeletons (frozen for eval is fine) ---
    src_model, tgt_model = build_models(
        args.src_model, args.tgt_model, args.projection_dim, freeze_base=True, device=device
    )

    # --- load projection heads ---
    src_head_path = os.path.join(args.output_dir, "src_proj.pt")
    tgt_head_path = os.path.join(args.output_dir, "tgt_proj.pt")
    src_model.proj.load_state_dict(torch.load(src_head_path, map_location=device))
    tgt_model.proj.load_state_dict(torch.load(tgt_head_path, map_location=device))

    # --- (optional) load LoRA adapters, if saved by train_embeddings.py ---
    # They would be in: {output_dir}/src_adapters and {output_dir}/tgt_adapters
    if args.use_lora or os.path.isdir(os.path.join(args.output_dir, "src_adapters")):
        src_model.base = maybe_load_lora_into_base(src_model.base, os.path.join(args.output_dir, "src_adapters"))
    if args.use_lora or os.path.isdir(os.path.join(args.output_dir, "tgt_adapters")):
        tgt_model.base = maybe_load_lora_into_base(tgt_model.base, os.path.join(args.output_dir, "tgt_adapters"))

    # --- evaluate with the chosen loss (defaults to InfoNCE) ---
    with torch.no_grad():
        test_loss = evaluate(
            src_model,
            tgt_model,
            test_loader,
            device,
            loss_type=args.loss_type,
            temperature=args.temperature,
            alpha=args.alpha,
            src_max_len=args.src_max_len,
            tgt_max_len=args.tgt_max_len,
        )

    print(f"Test {args.loss_type.upper()} loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
