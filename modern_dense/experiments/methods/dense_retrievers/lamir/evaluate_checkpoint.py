#!/usr/bin/env python3
"""Evaluate the completed supervised LAMIR checkpoint on a fixed split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def side_state(state: dict[str, torch.Tensor], side: str) -> dict[str, torch.Tensor]:
    prefix = f"{side}_encoder.encoder."
    result = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not result:
        raise RuntimeError(f"No {side} encoder parameters found")
    return result


def load_checkpoint_tokenizer(checkpoint_dir: Path, model_name: str):
    """Load the saved tokenizer, including across tokenizers JSON versions."""
    try:
        return AutoTokenizer.from_pretrained(checkpoint_dir), str(checkpoint_dir)
    except Exception as error:
        print(
            f"Could not load checkpoint tokenizer directly ({error}); "
            "trying a serializer-compatibility conversion",
            flush=True,
        )

    try:
        tokenizer_json = json.loads((checkpoint_dir / "tokenizer.json").read_text())
        components = list(tokenizer_json["pre_tokenizer"].get("pretokenizers", []))
        components.append(tokenizer_json.get("decoder", {}))
        for component in components:
            if component.get("type") != "Metaspace":
                continue
            # tokenizers >=0.20 serializes Metaspace with prepend_scheme/split;
            # tokenizers 0.15 uses the equivalent add_prefix_space field.
            prepend_scheme = component.pop("prepend_scheme", "always")
            component.pop("split", None)
            component["add_prefix_space"] = prepend_scheme != "never"
        backend = Tokenizer.from_str(json.dumps(tokenizer_json, ensure_ascii=False))
        config = json.loads((checkpoint_dir / "tokenizer_config.json").read_text())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=config.get("bos_token", "<s>"),
            cls_token=config.get("cls_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
            sep_token=config.get("sep_token", "</s>"),
            unk_token=config.get("unk_token", "<unk>"),
            pad_token=config.get("pad_token", "<pad>"),
            mask_token=config.get("mask_token", "<mask>"),
            model_max_length=config.get("model_max_length", 512),
        )
        return tokenizer, f"{checkpoint_dir} (compatibility-converted JSON)"
    except Exception as error:
        print(
            f"Could not compatibility-convert checkpoint tokenizer ({error}); "
            f"falling back to {model_name}",
            flush=True,
        )
        return AutoTokenizer.from_pretrained(model_name), model_name


@torch.inference_mode()
def encode(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_len: int,
) -> torch.Tensor:
    vectors = []
    model.to(device).eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        vectors.append(model(**batch, return_dict=True).last_hidden_state[:, 0].cpu())
    return F.normalize(torch.cat(vectors, dim=0).float(), p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv).dropna(
        subset=["Bahnaric", "Vietnamese"]
    ).reset_index(drop=True)
    source_texts = frame["Bahnaric"].astype(str).tolist()
    target_texts = frame["Vietnamese"].astype(str).tolist()
    checkpoint_path = args.checkpoint_dir / "model.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer, tokenizer_source = load_checkpoint_tokenizer(
        args.checkpoint_dir, args.model_name
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_model = AutoModel.from_pretrained(args.model_name)
    source_model.load_state_dict(side_state(state, "src"), strict=True)
    source = encode(
        source_texts, tokenizer, source_model, device, args.batch_size, args.max_len
    )
    source_model.to("cpu")
    del source_model
    torch.cuda.empty_cache()

    target_model = AutoModel.from_pretrained(args.model_name)
    target_model.load_state_dict(side_state(state, "tgt"), strict=True)
    target = encode(
        target_texts, tokenizer, target_model, device, args.batch_size, args.max_len
    )
    target_model.to("cpu")
    del target_model, state
    torch.cuda.empty_cache()

    scores = source.to(device) @ target.to(device).T
    k = min(args.topk, len(frame))
    top_scores, top_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    top_scores = top_scores.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    gold = np.arange(len(frame), dtype=np.int64)
    hits = top_indices == gold[:, None]
    ranks = np.full(len(frame), np.inf)
    found = hits.any(axis=1)
    ranks[found] = hits[found].argmax(axis=1) + 1

    metrics = {
        "method": "LAMIR supervised fine-tuning",
        "split": args.split,
        "num_queries": len(frame),
        "candidate_pool_size": len(frame),
        "acc@1": float(np.mean(ranks == 1)),
        "mrr@10": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "input_csv": str(args.input_csv),
        "input_sha256": sha256(args.input_csv),
        "pooling": "CLS",
        "similarity": "cosine",
        "tokenizer": tokenizer_source,
    }
    output = frame[["Bahnaric", "Vietnamese"]].copy()
    output["gold_rank"] = ranks
    output["topk_indices"] = ["|".join(map(str, row)) for row in top_indices]
    output["topk_scores"] = ["|".join(f"{value:.8g}" for value in row) for row in top_scores]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
