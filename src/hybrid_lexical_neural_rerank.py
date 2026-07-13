#!/usr/bin/env python3
"""Development-selected IBM1 + XLM-R LoRA hybrid reranking.

Two first-stage orders are compared on development data:

* ``neural_first``: XLM-R LoRA retrieves candidates, then IBM1 and neural
  scores are combined to rerank those candidates.
* ``ibm1_first``: IBM1 retrieves candidates from the full pool, then XLM-R
  LoRA and IBM1 scores are combined to rerank those candidates.

The IBM1 and XLM-R components must themselves already be selected on the
training-derived development split. Hybrid stage order, candidate top-k,
normalization, formula, interpolation weights, and length penalty are tuned on
``dev.csv``. Exactly one selected hybrid configuration may access ``test.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FAMILY = "hybrid"
EVALUATOR_SCRIPT = "src/hybrid_lexical_neural_rerank.py"
TRAINING_POLICY = "train_fit_only_no_dev_refit"
SELECTION_POLICY = (
    "development_selected_ibm1_and_xlmr_components_then_stage_order_and_"
    "hybrid_hyperparameters_selected_on_dev_then_single_test_evaluation"
)
REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid JSON from {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_text(
    text: str,
    lowercase: bool = True,
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


def tokenize(text: str) -> List[str]:
    text = str(text).strip()
    return text.split() if text else []


def read_parallel_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Parallel CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} contains no usable bilingual pairs")
    return frame


def split_pipe(cell: Any) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    return [part.strip() for part in text.split("|")] if text else []


# ---------------------------------------------------------------------------
# IBM Model 1
# ---------------------------------------------------------------------------


def build_candidate_sets(
    source_sentences: Sequence[Sequence[str]],
    target_sentences: Sequence[Sequence[str]],
    max_pairs_per_source_word: int = 200000,
) -> Dict[str, set[str]]:
    candidates: Dict[str, set[str]] = defaultdict(set)
    for source, target in zip(source_sentences, target_sentences):
        target_types = set(target)
        for source_word in set(source):
            if len(candidates[source_word]) < max_pairs_per_source_word:
                candidates[source_word].update(target_types)
    return candidates


def train_ibm1(
    source_sentences: Sequence[Sequence[str]],
    target_sentences: Sequence[Sequence[str]],
    num_iters: int,
    min_prob: float,
) -> Dict[str, Dict[str, float]]:
    candidate_sets = build_candidate_sets(source_sentences, target_sentences)
    table: Dict[str, Dict[str, float]] = {}
    for source_word, target_set in candidate_sets.items():
        targets = sorted(target_set)
        if targets:
            probability = 1.0 / float(len(targets))
            table[source_word] = {target: probability for target in targets}
    if not table:
        raise ValueError("IBM1 initialization produced an empty translation table")

    print(f"IBM1 candidate source vocabulary: {len(table)}")
    for iteration in range(1, num_iters + 1):
        counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        totals: Counter[str] = Counter()
        log_likelihood = 0.0
        events = 0

        for source, target in zip(source_sentences, target_sentences):
            if not source or not target:
                continue
            source_types = list(dict.fromkeys(source))
            for target_word in target:
                denominator = 0.0
                active: List[str] = []
                for source_word in source_types:
                    probability = table.get(source_word, {}).get(target_word, 0.0)
                    if probability > 0.0:
                        denominator += probability
                        active.append(source_word)
                if denominator <= 0.0:
                    continue
                log_likelihood += math.log(max(denominator, min_prob))
                events += 1
                for source_word in active:
                    delta = table[source_word][target_word] / denominator
                    counts[source_word][target_word] += delta
                    totals[source_word] += delta

        for source_word, target_counts in counts.items():
            denominator = float(totals[source_word])
            if denominator <= 0.0:
                continue
            for target_word, count in target_counts.items():
                table[source_word][target_word] = max(float(count) / denominator, min_prob)

        average = log_likelihood / max(1, events)
        print(f"IBM1 iteration {iteration}/{num_iters}: avg_log_likelihood={average:.6f}")
    return table


def lexical_score_forward(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    table: Dict[str, Dict[str, float]],
    smoothing: float,
) -> float:
    if not source_tokens or not target_tokens:
        return -1e9
    target_set = set(target_tokens)
    total = 0.0
    for source_word in source_tokens:
        translations = table.get(source_word)
        if not translations:
            total += math.log(smoothing)
            continue
        best = smoothing
        for target_word in target_set:
            best = max(best, translations.get(target_word, 0.0))
        total += math.log(max(best, smoothing))
    return total / float(max(1, len(source_tokens)))


def lexical_score(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    method: str,
    forward_table: Dict[str, Dict[str, float]],
    backward_table: Optional[Dict[str, Dict[str, float]]],
    smoothing: float,
) -> float:
    forward = lexical_score_forward(source_tokens, target_tokens, forward_table, smoothing)
    if method == "ibm1":
        return forward
    if method != "ibm1_sym" or backward_table is None:
        raise ValueError(f"Unsupported or incomplete IBM1 method: {method}")
    backward = lexical_score_forward(target_tokens, source_tokens, backward_table, smoothing)
    return 0.5 * (forward + backward)


def ibm_cache_key(train_path: Path, ibm_configuration: Dict[str, Any]) -> str:
    payload = {
        "train_sha256": sha256_file(train_path),
        "ibm_configuration": ibm_configuration,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:28]


def load_or_train_ibm(
    train_path: Path,
    ibm_configuration: Dict[str, Any],
    cache_dir: Path,
) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Dict[str, float]]], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ibm1_{ibm_cache_key(train_path, ibm_configuration)}.pkl"
    if cache_path.is_file():
        print(f"Loading cached IBM1 tables: {cache_path}")
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["forward"], payload.get("backward"), cache_path

    frame = read_parallel_csv(train_path)
    lowercase = bool(ibm_configuration["lowercase"])
    strip_accents = bool(ibm_configuration["strip_accents"])
    remove_punct = bool(ibm_configuration["remove_punct"])
    source_sentences = [
        tokenize(normalize_text(value, lowercase, strip_accents, remove_punct))
        for value in frame["Bahnaric"].astype(str)
    ]
    target_sentences = [
        tokenize(normalize_text(value, lowercase, strip_accents, remove_punct))
        for value in frame["Vietnamese"].astype(str)
    ]

    forward = train_ibm1(
        source_sentences,
        target_sentences,
        num_iters=int(ibm_configuration["ibm_iters"]),
        min_prob=float(ibm_configuration["smoothing"]),
    )
    backward: Optional[Dict[str, Dict[str, float]]] = None
    if ibm_configuration["method"] == "ibm1_sym":
        backward = train_ibm1(
            target_sentences,
            source_sentences,
            num_iters=int(ibm_configuration["ibm_iters"]),
            min_prob=float(ibm_configuration["smoothing"]),
        )

    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "configuration": ibm_configuration,
                "train_sha256": sha256_file(train_path),
                "forward": forward,
                "backward": backward,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(cache_path)
    print(f"Saved IBM1 cache: {cache_path}")
    return forward, backward, cache_path


# ---------------------------------------------------------------------------
# Selected component artifacts
# ---------------------------------------------------------------------------


def selected_component(manifest: Dict[str, Any], family: str) -> Dict[str, Any]:
    selected = manifest.get(family)
    if not isinstance(selected, dict):
        raise ValueError(
            f"Selection manifest has no {family!r} entry. Complete that family's "
            "development experiments and rebuild the selection manifest."
        )
    if not selected.get("configuration_name") or not isinstance(selected.get("configuration"), dict):
        raise ValueError(f"Malformed selected component entry for family {family!r}")
    return selected


def selected_ibm_configuration(selected: Dict[str, Any]) -> Dict[str, Any]:
    configuration = selected["configuration"]
    required = (
        "method",
        "ibm_iters",
        "smoothing",
        "strip_accents",
        "lowercase",
        "remove_punct",
    )
    missing = [key for key in required if key not in configuration]
    if missing:
        raise ValueError(f"Selected IBM1 configuration is missing: {missing}")
    return {
        "method": str(configuration["method"]),
        "ibm_iters": int(configuration["ibm_iters"]),
        "smoothing": float(configuration["smoothing"]),
        "strip_accents": bool(configuration["strip_accents"]),
        "lowercase": bool(configuration["lowercase"]),
        "remove_punct": bool(configuration["remove_punct"]),
        "base_length_penalty": float(configuration.get("length_penalty", 0.0)),
        "fit_policy": configuration.get("fit_policy", TRAINING_POLICY),
    }


def selected_dev_dir(selected: Dict[str, Any], label: str) -> Path:
    metrics_path = Path(str(selected.get("dev_metrics_path", "")))
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Selected {label} development metrics not found: {metrics_path}")
    return metrics_path.parent


def selected_test_dir(selected: Dict[str, Any], test_root: Path, family: str) -> Path:
    path = test_root / family / str(selected["configuration_name"])
    if not (path / "metrics.json").is_file():
        raise FileNotFoundError(
            f"Selected {family} test result not found under {path}. Run the central "
            f"selector with --evaluate_test --family {family} first."
        )
    return path


def required_artifact(directory: Path, filename: str, label: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} artifact not found: {path}")
    return path


# ---------------------------------------------------------------------------
# First-stage candidates and neural score matrix
# ---------------------------------------------------------------------------


@dataclass
class FirstStageCandidates:
    generator: str
    candidate_indices: np.ndarray
    first_stage_scores: np.ndarray
    available_topk: int
    predictions_path: str


def load_first_stage_candidates(
    predictions_path: Path,
    evaluation_frame: pd.DataFrame,
    generator: str,
) -> FirstStageCandidates:
    predictions = pd.read_csv(predictions_path)
    required = ("Bahnaric", "TopK_indices", "TopK_Scores")
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(
            f"{predictions_path} is missing {missing}. Regenerate the selected "
            f"{generator} prediction artifact with the updated evaluator."
        )
    if len(predictions) != len(evaluation_frame):
        raise ValueError(
            f"Prediction rows ({len(predictions)}) do not match evaluation rows "
            f"({len(evaluation_frame)}) for {predictions_path}"
        )

    expected_queries = evaluation_frame["Bahnaric"].astype(str).tolist()
    observed_queries = predictions["Bahnaric"].astype(str).tolist()
    mismatch_count = sum(a != b for a, b in zip(expected_queries, observed_queries))
    if mismatch_count:
        raise ValueError(
            f"{predictions_path} is not aligned with the evaluation CSV: "
            f"{mismatch_count} Bahnaric rows differ"
        )

    parsed_indices: List[List[int]] = []
    parsed_scores: List[List[float]] = []
    minimum_topk: Optional[int] = None
    candidate_pool_size = len(evaluation_frame)

    for row_number, row in predictions.iterrows():
        try:
            indices = [int(value) for value in split_pipe(row["TopK_indices"]) if value != ""]
            scores = [float(value) for value in split_pipe(row["TopK_Scores"]) if value != ""]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not parse TopK_indices/TopK_Scores at row {row_number} "
                f"in {predictions_path}"
            ) from exc
        if not indices or len(indices) != len(scores):
            raise ValueError(
                f"Invalid top-k fields at row {row_number} in {predictions_path}: "
                f"indices={len(indices)}, scores={len(scores)}"
            )
        if min(indices) < 0 or max(indices) >= candidate_pool_size:
            raise ValueError(
                f"Candidate index outside [0, {candidate_pool_size}) at row "
                f"{row_number} in {predictions_path}"
            )
        parsed_indices.append(indices)
        parsed_scores.append(scores)
        minimum_topk = len(indices) if minimum_topk is None else min(minimum_topk, len(indices))

    topk = int(minimum_topk or 0)
    if topk < 1:
        raise ValueError(f"No usable candidates found in {predictions_path}")
    indices_array = np.asarray([row[:topk] for row in parsed_indices], dtype=np.int64)
    scores_array = np.asarray([row[:topk] for row in parsed_scores], dtype=np.float64)
    return FirstStageCandidates(
        generator=generator,
        candidate_indices=indices_array,
        first_stage_scores=scores_array,
        available_topk=topk,
        predictions_path=str(predictions_path),
    )


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def neural_retrieval_settings(selected_neural: Dict[str, Any]) -> Tuple[str, int]:
    configuration = selected_neural["configuration"]
    retrieval = str(
        configuration.get(
            "retrieval",
            "csls" if configuration.get("use_csls") else "cosine",
        )
    )
    if retrieval not in {"cosine", "csls"}:
        raise ValueError(f"Unsupported selected neural retrieval mode: {retrieval}")
    return retrieval, int(configuration.get("csls_k", 10))


def neural_score_cache_key(
    embeddings_path: Path,
    retrieval: str,
    csls_k: int,
) -> str:
    payload = {
        "embeddings_sha256": sha256_file(embeddings_path),
        "retrieval": retrieval,
        "csls_k": int(csls_k),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:28]


def compute_neural_score_matrix(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    retrieval: str,
    csls_k: int,
) -> np.ndarray:
    source = l2_normalize_rows(source_embeddings)
    target = l2_normalize_rows(target_embeddings)
    scores = np.asarray(source @ target.T, dtype=np.float32)
    if retrieval == "cosine":
        return scores
    if retrieval != "csls":
        raise ValueError(f"Unsupported retrieval mode: {retrieval}")
    if source.shape[0] <= 1 or target.shape[0] <= 1:
        return scores
    effective_k = int(max(1, min(csls_k, source.shape[0] - 1, target.shape[0] - 1)))
    source_density = np.partition(scores, -effective_k, axis=1)[:, -effective_k:].mean(axis=1)
    target_density = np.partition(scores, -effective_k, axis=0)[-effective_k:, :].mean(axis=0)
    scores *= 2.0
    scores -= source_density[:, None]
    scores -= target_density[None, :]
    return scores


def load_or_build_neural_score_matrix(
    embeddings_path: Path,
    selected_neural: Dict[str, Any],
    cache_dir: Path,
    expected_rows: int,
) -> Tuple[np.ndarray, Path]:
    retrieval, csls_k = neural_retrieval_settings(selected_neural)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"neural_scores_{neural_score_cache_key(embeddings_path, retrieval, csls_k)}.npy"
    )
    if cache_path.is_file():
        matrix = np.load(cache_path, mmap_mode="r")
        if matrix.shape == (expected_rows, expected_rows):
            print(f"Loaded neural score cache: {cache_path}")
            return matrix, cache_path
        print(f"Ignoring neural score cache with unexpected shape: {cache_path}")

    with np.load(embeddings_path) as data:
        if "source" not in data or "target" not in data:
            raise ValueError(
                f"{embeddings_path} must contain source and target retrieval embeddings"
            )
        source = np.asarray(data["source"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
    if source.shape[0] != expected_rows or target.shape[0] != expected_rows:
        raise ValueError(
            f"Retrieval embedding rows do not match evaluation rows: source={source.shape}, "
            f"target={target.shape}, expected={expected_rows}"
        )

    print(f"Computing full {retrieval} neural score matrix for {expected_rows} rows")
    scores = compute_neural_score_matrix(source, target, retrieval, csls_k)
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, scores)
    os.replace(temporary, cache_path)
    print(f"Saved neural score cache: {cache_path}")
    return np.load(cache_path, mmap_mode="r"), cache_path


# ---------------------------------------------------------------------------
# Hybrid feature preparation and scoring
# ---------------------------------------------------------------------------


@dataclass
class PreparedFeatures:
    candidate_generator: str
    candidate_indices: np.ndarray
    candidate_texts: List[List[str]]
    first_stage_scores: np.ndarray
    neural_raw: np.ndarray
    reciprocal_rank: np.ndarray
    linear_rank: np.ndarray
    ibm_base: np.ndarray
    length_difference: np.ndarray
    levenshtein: np.ndarray
    valid_mask: np.ndarray
    gold_mask: np.ndarray
    raw_sources: List[str]
    raw_targets: List[str]
    first_stage_predictions_path: str


def neural_rank_features(
    row_scores: np.ndarray,
    candidate_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    row_scores = np.asarray(row_scores, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_scores = row_scores[candidate_indices]
    all_indices = np.arange(row_scores.shape[0], dtype=np.int64)
    greater = np.sum(row_scores[:, None] > candidate_scores[None, :], axis=0)
    tied_before = np.sum(
        (row_scores[:, None] == candidate_scores[None, :])
        & (all_indices[:, None] < candidate_indices[None, :]),
        axis=0,
    )
    ranks = 1.0 + greater.astype(np.float64) + tied_before.astype(np.float64)
    reciprocal = 1.0 / ranks
    linear = 1.0 - (ranks - 1.0) / float(max(1, row_scores.shape[0] - 1))
    return reciprocal, linear


def prepare_features(
    frame: pd.DataFrame,
    first_stage: FirstStageCandidates,
    neural_score_matrix: np.ndarray,
    ibm_configuration: Dict[str, Any],
    forward_table: Dict[str, Dict[str, float]],
    backward_table: Optional[Dict[str, Dict[str, float]]],
) -> PreparedFeatures:
    number_of_rows = len(frame)
    topk = first_stage.available_topk
    if first_stage.candidate_indices.shape != (number_of_rows, topk):
        raise ValueError("First-stage candidate matrix shape does not match evaluation data")

    neural_raw = np.zeros((number_of_rows, topk), dtype=np.float64)
    reciprocal_rank = np.zeros((number_of_rows, topk), dtype=np.float64)
    linear_rank = np.zeros((number_of_rows, topk), dtype=np.float64)
    ibm_base = np.full((number_of_rows, topk), -1e9, dtype=np.float64)
    length_difference = np.zeros((number_of_rows, topk), dtype=np.float64)
    levenshtein = np.zeros((number_of_rows, topk), dtype=np.float64)
    valid_mask = np.ones((number_of_rows, topk), dtype=bool)
    gold_mask = np.zeros((number_of_rows, topk), dtype=bool)

    raw_sources = frame["Bahnaric"].astype(str).tolist()
    raw_targets = frame["Vietnamese"].astype(str).tolist()
    lowercase = bool(ibm_configuration["lowercase"])
    strip_accents = bool(ibm_configuration["strip_accents"])
    remove_punct = bool(ibm_configuration["remove_punct"])
    method = str(ibm_configuration["method"])
    smoothing = float(ibm_configuration["smoothing"])
    candidate_texts: List[List[str]] = []

    for row_index, raw_source in enumerate(raw_sources):
        indices = first_stage.candidate_indices[row_index]
        texts = [raw_targets[int(index)] for index in indices]
        candidate_texts.append(texts)
        neural_raw[row_index] = np.asarray(neural_score_matrix[row_index, indices], dtype=np.float64)
        reciprocal, linear = neural_rank_features(neural_score_matrix[row_index], indices)
        reciprocal_rank[row_index] = reciprocal
        linear_rank[row_index] = linear

        normalized_source = normalize_text(
            raw_source,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
        )
        source_tokens = tokenize(normalized_source)
        for rank_index, (candidate_index, candidate_text) in enumerate(zip(indices, texts)):
            normalized_candidate = normalize_text(
                candidate_text,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
            candidate_tokens = tokenize(normalized_candidate)
            ibm_base[row_index, rank_index] = lexical_score(
                source_tokens,
                candidate_tokens,
                method=method,
                forward_table=forward_table,
                backward_table=backward_table,
                smoothing=smoothing,
            )
            length_difference[row_index, rank_index] = abs(
                len(source_tokens) - len(candidate_tokens)
            ) / float(max(len(source_tokens), len(candidate_tokens), 1))
            levenshtein[row_index, rank_index] = float(
                SequenceMatcher(None, normalized_source, normalized_candidate).ratio()
            )
            gold_mask[row_index, rank_index] = int(candidate_index) == row_index

        if row_index % 250 == 0:
            print(
                f"Prepared {first_stage.generator} hybrid features for "
                f"{row_index}/{number_of_rows} queries"
            )

    return PreparedFeatures(
        candidate_generator=first_stage.generator,
        candidate_indices=first_stage.candidate_indices,
        candidate_texts=candidate_texts,
        first_stage_scores=first_stage.first_stage_scores,
        neural_raw=neural_raw,
        reciprocal_rank=reciprocal_rank,
        linear_rank=linear_rank,
        ibm_base=ibm_base,
        length_difference=length_difference,
        levenshtein=levenshtein,
        valid_mask=valid_mask,
        gold_mask=gold_mask,
        raw_sources=raw_sources,
        raw_targets=raw_targets,
        first_stage_predictions_path=first_stage.predictions_path,
    )


def normalize_rows(values: np.ndarray, valid_mask: np.ndarray, mode: str) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    for row_index in range(values.shape[0]):
        mask = valid_mask[row_index]
        row = values[row_index, mask].astype(np.float64)
        if row.size == 0:
            continue
        if mode == "none":
            normalized = row
        elif mode == "minmax":
            low, high = float(np.min(row)), float(np.max(row))
            normalized = np.zeros_like(row) if abs(high - low) < 1e-12 else (row - low) / (high - low)
        elif mode == "zscore":
            mean, standard_deviation = float(np.mean(row)), float(np.std(row))
            normalized = np.zeros_like(row) if standard_deviation < 1e-12 else (row - mean) / standard_deviation
        elif mode == "rank":
            order = np.argsort(-row, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(row))
            normalized = 1.0 - ranks.astype(np.float64) / float(max(1, len(row) - 1))
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        output[row_index, mask] = normalized
    return output


def positive_unit_interval(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.clip(normalize_rows(values, valid_mask, "minmax"), 1e-8, 1.0)


def hybrid_scores(
    neural_values: np.ndarray,
    ibm_values: np.ndarray,
    levenshtein_values: np.ndarray,
    valid_mask: np.ndarray,
    normalization: str,
    formula: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    neural = normalize_rows(neural_values, valid_mask, normalization)
    ibm = normalize_rows(ibm_values, valid_mask, normalization)
    levenshtein = normalize_rows(levenshtein_values, valid_mask, normalization)

    if formula == "weighted_sum":
        scores = alpha * neural + beta * ibm + gamma * levenshtein
    elif formula == "weighted_product":
        scores = (
            np.power(positive_unit_interval(neural, valid_mask), alpha)
            * np.power(positive_unit_interval(ibm, valid_mask), beta)
            * np.power(positive_unit_interval(levenshtein, valid_mask), gamma)
        )
    else:
        raise ValueError(f"Unknown reranking formula: {formula}")
    return np.where(valid_mask, scores, -np.inf)


def metrics_from_ranks(
    ranks: np.ndarray,
    effective_topk: int,
    eval_ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "Top1_acc": float(np.mean(ranks == 1.0)),
        "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
    }
    for k in eval_ks:
        k_effective = int(max(1, min(int(k), effective_topk)))
        hit = float(np.mean(ranks <= k_effective))
        metrics[f"Hit@{k}"] = hit
        metrics[f"Recall@{k}"] = hit
        metrics[f"Precision@{k}"] = hit / float(k_effective)
    return metrics


def evaluate_scores(
    scores: np.ndarray,
    gold_mask: np.ndarray,
    valid_mask: np.ndarray,
    candidate_topk: int,
    eval_ks: Sequence[int],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    effective_topk = int(min(candidate_topk, scores.shape[1]))
    restricted_scores = np.where(
        valid_mask[:, :effective_topk],
        scores[:, :effective_topk],
        -np.inf,
    )
    restricted_gold = gold_mask[:, :effective_topk]
    ordering = np.argsort(-restricted_scores, axis=1, kind="mergesort")
    ranks = np.full(scores.shape[0], np.inf, dtype=np.float64)
    for row_index in range(scores.shape[0]):
        positions = np.where(restricted_gold[row_index, ordering[row_index]])[0]
        if positions.size:
            ranks[row_index] = float(positions[0] + 1)
    return metrics_from_ranks(ranks, effective_topk, eval_ks), ranks, ordering


def evaluate_first_stage(
    features: PreparedFeatures,
    candidate_topk: int,
    eval_ks: Sequence[int],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    effective_topk = int(min(candidate_topk, features.gold_mask.shape[1]))
    ranks = np.full(features.gold_mask.shape[0], np.inf, dtype=np.float64)
    for row_index in range(features.gold_mask.shape[0]):
        positions = np.where(features.gold_mask[row_index, :effective_topk])[0]
        if positions.size:
            ranks[row_index] = float(positions[0] + 1)
    ordering = np.tile(np.arange(effective_topk, dtype=np.int64), (len(ranks), 1))
    return metrics_from_ranks(ranks, effective_topk, eval_ks), ranks, ordering


def parse_weight_specs(specifications: Sequence[str]) -> List[Tuple[float, float, float]]:
    weights: List[Tuple[float, float, float]] = []
    for specification in specifications:
        parts = specification.split(",")
        if len(parts) != 3:
            raise ValueError(f"Weight specification must be alpha,beta,gamma: {specification!r}")
        alpha, beta, gamma = (float(part) for part in parts)
        if min(alpha, beta, gamma) < 0.0:
            raise ValueError("Hybrid weights must be non-negative")
        if alpha + beta + gamma <= 0.0:
            raise ValueError("At least one hybrid weight must be positive")
        weights.append((alpha, beta, gamma))
    return weights


def float_token(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def configuration_name(configuration: Dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()[:10]
    return (
        f"hybrid_{configuration['candidate_generator']}_"
        f"k{configuration['candidate_topk']}_"
        f"{configuration['neural_score_mode']}_{configuration['normalization']}_"
        f"{configuration['formula']}_"
        f"a{float_token(configuration['alpha'])}_"
        f"b{float_token(configuration['beta'])}_"
        f"g{float_token(configuration['gamma'])}_"
        f"lp{float_token(configuration['length_penalty'])}_{digest}"
    )


def build_configuration(
    selected_ibm: Dict[str, Any],
    selected_neural: Dict[str, Any],
    ibm_configuration: Dict[str, Any],
    candidate_generator: str,
    candidate_topk: int,
    neural_score_mode: str,
    normalization: str,
    formula: str,
    alpha: float,
    beta: float,
    gamma: float,
    length_penalty: float,
) -> Dict[str, Any]:
    if candidate_generator not in {"neural_first", "ibm1_first"}:
        raise ValueError(f"Unknown candidate generator: {candidate_generator}")
    return {
        "candidate_generator": candidate_generator,
        "first_stage_family": (
            "xlmr_lora_projection" if candidate_generator == "neural_first" else "ibm1"
        ),
        "second_stage_family": (
            "ibm1" if candidate_generator == "neural_first" else "xlmr_lora_projection"
        ),
        "ibm1_family": "ibm1",
        "ibm1_configuration_name": selected_ibm["configuration_name"],
        "ibm1_configuration": ibm_configuration,
        "neural_family": "xlmr_lora_projection",
        "neural_configuration_name": selected_neural["configuration_name"],
        "neural_configuration": selected_neural["configuration"],
        "candidate_topk": int(candidate_topk),
        "neural_score_mode": neural_score_mode,
        "normalization": normalization,
        "formula": formula,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "length_penalty": float(length_penalty),
        "training_policy": TRAINING_POLICY,
        "selection_policy": SELECTION_POLICY,
    }


def build_evaluation_cli_args(args: argparse.Namespace, configuration: Dict[str, Any]) -> List[str]:
    return [
        "--mode",
        "evaluate",
        "--train_csv",
        str(args.train_csv),
        "--test_root",
        str(args.test_root),
        "--ibm_cache_dir",
        str(args.ibm_cache_dir),
        "--neural_score_cache_dir",
        str(args.neural_score_cache_dir),
        "--ibm1_family",
        str(args.ibm1_family),
        "--neural_family",
        str(args.neural_family),
        "--candidate_generator",
        str(configuration["candidate_generator"]),
        "--candidate_topk",
        str(configuration["candidate_topk"]),
        "--neural_score_mode",
        str(configuration["neural_score_mode"]),
        "--normalization",
        str(configuration["normalization"]),
        "--formula",
        str(configuration["formula"]),
        "--alpha",
        str(configuration["alpha"]),
        "--beta",
        str(configuration["beta"]),
        "--gamma",
        str(configuration["gamma"]),
        "--length_penalty",
        str(configuration["length_penalty"]),
        "--eval_ks",
        *[str(value) for value in args.eval_ks],
    ]


def choose_neural_matrix(features: PreparedFeatures, mode: str) -> np.ndarray:
    if mode == "raw":
        return features.neural_raw
    if mode == "reciprocal_rank":
        return features.reciprocal_rank
    if mode == "linear_rank":
        return features.linear_rank
    raise ValueError(f"Unknown neural score mode: {mode}")


def rounded_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in metrics.items()}


def write_predictions(
    path: Path,
    features: PreparedFeatures,
    scores: np.ndarray,
    ordering: np.ndarray,
    final_ranks: np.ndarray,
    first_stage_ranks: np.ndarray,
    candidate_topk: int,
    configuration: Dict[str, Any],
) -> None:
    effective_topk = int(min(candidate_topk, scores.shape[1]))
    rows: List[Dict[str, Any]] = []
    for row_index in range(scores.shape[0]):
        first_indices = features.candidate_indices[row_index, :effective_topk]
        first_texts = features.candidate_texts[row_index][:effective_topk]
        first_scores = features.first_stage_scores[row_index, :effective_topk]
        order = ordering[row_index]
        reranked_indices = [int(first_indices[index]) for index in order]
        reranked_texts = [first_texts[index] for index in order]
        reranked_scores = [float(scores[row_index, index]) for index in order]
        rows.append(
            {
                "Bahnaric": features.raw_sources[row_index],
                "Gold_VN": features.raw_targets[row_index],
                "candidate_generator": features.candidate_generator,
                "Predicted_VN": reranked_texts[0] if reranked_texts else "",
                "First_stage_gold_rank": (
                    None if not np.isfinite(first_stage_ranks[row_index])
                    else int(first_stage_ranks[row_index])
                ),
                "Gold_rank": (
                    None if not np.isfinite(final_ranks[row_index])
                    else int(final_ranks[row_index])
                ),
                "TopK_FirstStage": "|".join(first_texts),
                "TopK_FirstStage_indices": "|".join(str(int(value)) for value in first_indices),
                "TopK_FirstStage_Scores": "|".join(f"{float(value):.10g}" for value in first_scores),
                "TopK_Reranked": "|".join(reranked_texts),
                "TopK_Reranked_indices": "|".join(str(value) for value in reranked_indices),
                "TopK_Reranked_Scores": "|".join(f"{value:.10g}" for value in reranked_scores),
                "configuration_name": configuration_name(configuration),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def component_context(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    manifest_path = Path(args.selection_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    selected_ibm = selected_component(manifest, args.ibm1_family)
    selected_neural = selected_component(manifest, args.neural_family)
    ibm_configuration = selected_ibm_configuration(selected_ibm)
    if ibm_configuration.get("fit_policy") != TRAINING_POLICY:
        raise ValueError(
            "Selected IBM1 component does not declare train_fit_only_no_dev_refit"
        )
    return manifest, selected_ibm, selected_neural, ibm_configuration


def split_artifact_directories(
    args: argparse.Namespace,
    selected_ibm: Dict[str, Any],
    selected_neural: Dict[str, Any],
) -> Tuple[Path, Path]:
    if args.split_name == "dev":
        return (
            selected_dev_dir(selected_ibm, "IBM1"),
            selected_dev_dir(selected_neural, "XLM-R"),
        )
    if args.split_name == "test":
        test_root = Path(args.test_root)
        return (
            selected_test_dir(selected_ibm, test_root, args.ibm1_family),
            selected_test_dir(selected_neural, test_root, args.neural_family),
        )
    raise ValueError(f"Unsupported split: {args.split_name}")


def prepare_split_features(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    selected_ibm: Dict[str, Any],
    selected_neural: Dict[str, Any],
    ibm_configuration: Dict[str, Any],
    requested_generators: Sequence[str],
) -> Tuple[Dict[str, PreparedFeatures], Path, Path]:
    forward, backward, ibm_cache_path = load_or_train_ibm(
        Path(args.train_csv),
        ibm_configuration,
        Path(args.ibm_cache_dir),
    )
    ibm_dir, neural_dir = split_artifact_directories(args, selected_ibm, selected_neural)
    embeddings_path = required_artifact(
        neural_dir,
        "retrieval_embeddings.npz",
        "XLM-R retrieval embeddings",
    )
    neural_score_matrix, neural_score_cache_path = load_or_build_neural_score_matrix(
        embeddings_path,
        selected_neural,
        Path(args.neural_score_cache_dir),
        len(frame),
    )

    features: Dict[str, PreparedFeatures] = {}
    for generator in requested_generators:
        if generator == "neural_first":
            predictions_path = required_artifact(
                neural_dir,
                "sentence_predictions.csv",
                "selected XLM-R predictions",
            )
        elif generator == "ibm1_first":
            predictions_path = required_artifact(
                ibm_dir,
                "sentence_predictions.csv",
                "selected IBM1 predictions",
            )
        else:
            raise ValueError(f"Unknown candidate generator: {generator}")
        first_stage = load_first_stage_candidates(predictions_path, frame, generator)
        features[generator] = prepare_features(
            frame,
            first_stage,
            neural_score_matrix,
            ibm_configuration,
            forward,
            backward,
        )
    return features, ibm_cache_path, neural_score_cache_path


def build_result_record(
    args: argparse.Namespace,
    input_path: Path,
    configuration: Dict[str, Any],
    features: PreparedFeatures,
    final_metrics: Dict[str, float],
    first_stage_metrics: Dict[str, float],
    ibm_cache_path: Path,
    neural_score_cache_path: Path,
) -> Dict[str, Any]:
    final_rounded = rounded_metrics(final_metrics)
    first_rounded = rounded_metrics(first_stage_metrics)
    gains = {
        "Top1_acc": round(
            float(final_metrics["Top1_acc"] - first_stage_metrics["Top1_acc"]), 4
        ),
        "MRR": round(float(final_metrics["MRR"] - first_stage_metrics["MRR"]), 4),
    }
    candidate_topk = int(configuration["candidate_topk"])
    recall_key = f"Recall@{candidate_topk}"
    record: Dict[str, Any] = dict(final_rounded)
    record.update(
        {
            "schema_version": 2,
            "family": FAMILY,
            "split": args.split_name,
            "configuration_name": configuration_name(configuration),
            "configuration": configuration,
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": build_evaluation_cli_args(args, configuration),
            "method": "hybrid_ibm1_xlmr_lora_bidirectional_stage_order",
            "candidate_generator": configuration["candidate_generator"],
            "first_stage_family": configuration["first_stage_family"],
            "second_stage_family": configuration["second_stage_family"],
            "first_stage_metrics": first_rounded,
            "final_metrics": final_rounded,
            "reranking_gain": gains,
            "first_stage_top1_acc": first_rounded["Top1_acc"],
            "first_stage_mrr": first_rounded["MRR"],
            "first_stage_recall_at_k": first_rounded.get(recall_key),
            "final_top1_acc": final_rounded["Top1_acc"],
            "final_mrr": final_rounded["MRR"],
            "rerank_top1_gain": gains["Top1_acc"],
            "rerank_mrr_gain": gains["MRR"],
            "input_csv": str(input_path),
            "input_sha256": sha256_file(input_path),
            "train_csv": str(args.train_csv),
            "train_sha256": sha256_file(Path(args.train_csv)),
            "first_stage_predictions": features.first_stage_predictions_path,
            "ibm_cache": str(ibm_cache_path),
            "neural_score_cache": str(neural_score_cache_path),
            "num_queries": int(len(features.raw_sources)),
            "full_candidate_pool_size": int(len(features.raw_targets)),
            "candidate_pool_size": candidate_topk,
            "available_first_stage_topk": int(features.candidate_indices.shape[1]),
            "gold_match_mode": "candidate_index",
            "selection_policy": SELECTION_POLICY,
            "training_policy": TRAINING_POLICY,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Development tuning and selected evaluation
# ---------------------------------------------------------------------------


def tune_dev(args: argparse.Namespace) -> None:
    if args.split_name != "dev":
        raise ValueError("tune_dev mode requires --split_name dev")
    _, selected_ibm, selected_neural, ibm_configuration = component_context(args)
    input_path = Path(args.input_csv)
    frame = read_parallel_csv(input_path)
    generators = list(dict.fromkeys(args.candidate_generators))
    features_by_generator, ibm_cache_path, neural_score_cache_path = prepare_split_features(
        args,
        frame,
        selected_ibm,
        selected_neural,
        ibm_configuration,
        generators,
    )

    weights = parse_weight_specs(args.weights)
    score_modes = list(dict.fromkeys(args.neural_score_modes))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_sha256 = sha256_file(input_path)

    expected_names: List[str] = []
    summaries: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[float, float, float, str]] = None
    best_payload: Optional[
        Tuple[Dict[str, Any], PreparedFeatures, np.ndarray, np.ndarray, np.ndarray]
    ] = None

    for generator in generators:
        features = features_by_generator[generator]
        candidate_topks = sorted(
            {
                int(value)
                for value in args.candidate_topks
                if 0 < int(value) <= features.candidate_indices.shape[1]
            }
        )
        if not candidate_topks:
            raise ValueError(
                f"No requested candidate top-k is available for {generator}; "
                f"artifact provides top-k={features.candidate_indices.shape[1]}"
            )

        for candidate_topk in candidate_topks:
            first_metrics, first_ranks, _ = evaluate_first_stage(
                features, candidate_topk, args.eval_ks
            )
            for score_mode in score_modes:
                neural_values = choose_neural_matrix(features, score_mode)
                for normalization in args.normalizations:
                    for formula in args.formulas:
                        for length_penalty in args.length_penalties:
                            ibm_values = (
                                features.ibm_base
                                - float(length_penalty) * features.length_difference
                            )
                            for alpha, beta, gamma in weights:
                                configuration = build_configuration(
                                    selected_ibm,
                                    selected_neural,
                                    ibm_configuration,
                                    generator,
                                    candidate_topk,
                                    score_mode,
                                    normalization,
                                    formula,
                                    alpha,
                                    beta,
                                    gamma,
                                    length_penalty,
                                )
                                name = configuration_name(configuration)
                                expected_names.append(name)
                                scores = hybrid_scores(
                                    neural_values,
                                    ibm_values,
                                    features.levenshtein,
                                    features.valid_mask,
                                    normalization,
                                    formula,
                                    alpha,
                                    beta,
                                    gamma,
                                )
                                final_metrics, final_ranks, ordering = evaluate_scores(
                                    scores,
                                    features.gold_mask,
                                    features.valid_mask,
                                    candidate_topk,
                                    args.eval_ks,
                                )
                                record = build_result_record(
                                    args,
                                    input_path,
                                    configuration,
                                    features,
                                    final_metrics,
                                    first_metrics,
                                    ibm_cache_path,
                                    neural_score_cache_path,
                                )
                                record["split"] = "dev"
                                record["input_sha256"] = input_sha256
                                metrics_path = output_root / name / "metrics.json"
                                should_write = True
                                if metrics_path.is_file() and not args.overwrite:
                                    existing = load_json(metrics_path)
                                    should_write = not (
                                        existing.get("input_sha256") == input_sha256
                                        and canonical_json(existing.get("configuration"))
                                        == canonical_json(configuration)
                                    )
                                if should_write:
                                    atomic_write_json(metrics_path, record)

                                summary = {
                                    "configuration_name": name,
                                    "candidate_generator": generator,
                                    "candidate_topk": candidate_topk,
                                    "first_stage_Top1_acc": record["first_stage_top1_acc"],
                                    "first_stage_MRR": record["first_stage_mrr"],
                                    "first_stage_Recall_at_k": record["first_stage_recall_at_k"],
                                    "Top1_acc": record["Top1_acc"],
                                    "MRR": record["MRR"],
                                    "Recall@5": record.get("Recall@5"),
                                    "Recall@10": record.get("Recall@10"),
                                    "rerank_top1_gain": record["rerank_top1_gain"],
                                    "rerank_mrr_gain": record["rerank_mrr_gain"],
                                    "neural_score_mode": score_mode,
                                    "normalization": normalization,
                                    "formula": formula,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "gamma": gamma,
                                    "length_penalty": length_penalty,
                                }
                                summaries.append(summary)
                                candidate_key = (
                                    -float(record["Top1_acc"]),
                                    -float(record["MRR"]),
                                    -float(record.get("Recall@5", -1.0)),
                                    name,
                                )
                                if best_key is None or candidate_key < best_key:
                                    best_key = candidate_key
                                    best_payload = (
                                        configuration,
                                        features,
                                        scores,
                                        final_ranks,
                                        first_ranks,
                                    )

    if best_payload is None:
        raise RuntimeError("No hybrid development configurations were evaluated")

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["Top1_acc", "MRR", "Recall@5", "configuration_name"],
        ascending=[False, False, False, True],
    )
    summary_frame.to_csv(output_root / "hybrid_dev_grid.csv", index=False)

    comparison = (
        summary_frame.sort_values(
            ["candidate_generator", "candidate_topk", "Top1_acc", "MRR", "configuration_name"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["candidate_generator", "candidate_topk"], as_index=False)
        .first()
    )
    comparison.to_csv(output_root / "stage_order_comparison.csv", index=False)

    best_configuration, best_features, best_scores, best_final_ranks, best_first_ranks = best_payload
    best_name = configuration_name(best_configuration)
    _, _, best_ordering = evaluate_scores(
        best_scores,
        best_features.gold_mask,
        best_features.valid_mask,
        int(best_configuration["candidate_topk"]),
        args.eval_ks,
    )
    write_predictions(
        output_root / best_name / "sentence_predictions.csv",
        best_features,
        best_scores,
        best_ordering,
        best_final_ranks,
        best_first_ranks,
        int(best_configuration["candidate_topk"]),
        best_configuration,
    )

    search_manifest = {
        "schema_version": 2,
        "family": FAMILY,
        "input_csv": str(input_path),
        "input_sha256": input_sha256,
        "train_csv": str(args.train_csv),
        "selected_ibm1_configuration_name": selected_ibm["configuration_name"],
        "selected_neural_configuration_name": selected_neural["configuration_name"],
        "candidate_generators": generators,
        "candidate_topks": [int(value) for value in args.candidate_topks],
        "neural_score_modes": score_modes,
        "normalizations": list(args.normalizations),
        "formulas": list(args.formulas),
        "length_penalties": [float(value) for value in args.length_penalties],
        "weights": [list(weight) for weight in weights],
        "expected_configuration_count": len(expected_names),
        "expected_configuration_names": sorted(expected_names),
        "best_by_declared_order": best_name,
        "stage_order_comparison_csv": str(output_root / "stage_order_comparison.csv"),
    }
    atomic_write_json(output_root / "search_manifest.json", search_manifest)

    if args.prune_stale:
        expected_set = set(expected_names)
        for child in output_root.iterdir():
            if child.is_dir() and child.name not in expected_set:
                print(f"Removing stale hybrid development result: {child}")
                shutil.rmtree(child)

    print(f"Completed {len(expected_names)} hybrid development configurations")
    print(f"Best development configuration: {best_name}")
    print(f"Best candidate generator: {best_configuration['candidate_generator']}")
    print(f"Saved grid to {output_root / 'hybrid_dev_grid.csv'}")
    print(f"Saved order comparison to {output_root / 'stage_order_comparison.csv'}")


def validate_hybrid_test_authorization(
    args: argparse.Namespace,
    configuration: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    if args.split_name != "test":
        return
    selected = manifest.get(FAMILY)
    if not isinstance(selected, dict):
        raise ValueError("Selection manifest has no development-selected hybrid entry")
    if selected.get("configuration_name") != args.configuration_name:
        raise ValueError(
            f"Unauthorized hybrid test configuration {args.configuration_name!r}; "
            f"development selected {selected.get('configuration_name')!r}"
        )
    if canonical_json(selected.get("configuration")) != canonical_json(configuration):
        raise ValueError(
            "Hybrid test arguments, stage order, or base components do not match "
            "the development-selected hybrid configuration"
        )


def evaluate_one(args: argparse.Namespace) -> None:
    manifest, selected_ibm, selected_neural, ibm_configuration = component_context(args)
    if not args.configuration_name:
        raise ValueError("evaluate mode requires --configuration_name")

    configuration = build_configuration(
        selected_ibm,
        selected_neural,
        ibm_configuration,
        args.candidate_generator,
        args.candidate_topk,
        args.neural_score_mode,
        args.normalization,
        args.formula,
        args.alpha,
        args.beta,
        args.gamma,
        args.length_penalty,
    )
    expected_name = configuration_name(configuration)
    if expected_name != args.configuration_name:
        raise ValueError(
            "Configuration name does not match the supplied hybrid arguments. "
            f"Expected {expected_name!r}."
        )
    validate_hybrid_test_authorization(args, configuration, manifest)

    input_path = Path(args.input_csv)
    frame = read_parallel_csv(input_path)
    features_by_generator, ibm_cache_path, neural_score_cache_path = prepare_split_features(
        args,
        frame,
        selected_ibm,
        selected_neural,
        ibm_configuration,
        [args.candidate_generator],
    )
    features = features_by_generator[args.candidate_generator]
    if args.candidate_topk > features.candidate_indices.shape[1]:
        raise ValueError(
            f"Selected candidate_topk={args.candidate_topk} exceeds available "
            f"top-k={features.candidate_indices.shape[1]} for {args.candidate_generator}"
        )

    first_metrics, first_ranks, _ = evaluate_first_stage(
        features, args.candidate_topk, args.eval_ks
    )
    neural_values = choose_neural_matrix(features, args.neural_score_mode)
    ibm_values = features.ibm_base - float(args.length_penalty) * features.length_difference
    scores = hybrid_scores(
        neural_values,
        ibm_values,
        features.levenshtein,
        features.valid_mask,
        args.normalization,
        args.formula,
        args.alpha,
        args.beta,
        args.gamma,
    )
    final_metrics, final_ranks, ordering = evaluate_scores(
        scores,
        features.gold_mask,
        features.valid_mask,
        args.candidate_topk,
        args.eval_ks,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output_dir / "sentence_predictions.csv",
        features,
        scores,
        ordering,
        final_ranks,
        first_ranks,
        args.candidate_topk,
        configuration,
    )
    record = build_result_record(
        args,
        input_path,
        configuration,
        features,
        final_metrics,
        first_metrics,
        ibm_cache_path,
        neural_score_cache_path,
    )
    record["configuration_name"] = args.configuration_name
    atomic_write_json(output_dir / "metrics.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


def main(args: argparse.Namespace) -> None:
    if Path(args.train_csv).resolve() == Path(args.input_csv).resolve():
        raise ValueError("Training and evaluation CSVs must be different")
    if args.mode == "tune_dev":
        tune_dev(args)
    elif args.mode == "evaluate":
        evaluate_one(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Development-selected IBM1/XLM-R hybrid reranking with neural-first "
            "and IBM1-first stage-order comparison"
        )
    )
    parser.add_argument("--mode", choices=["tune_dev", "evaluate"], required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--configuration_name", default=None)

    parser.add_argument("--ibm1_family", default="ibm1")
    parser.add_argument("--neural_family", default="xlmr_lora_projection")
    parser.add_argument("--test_root", default="results/test")
    parser.add_argument("--ibm_cache_dir", default="results/cache/hybrid_ibm1")
    parser.add_argument(
        "--neural_score_cache_dir",
        default="results/cache/hybrid_neural_scores",
    )

    parser.add_argument(
        "--candidate_generators",
        nargs="+",
        choices=["neural_first", "ibm1_first"],
        default=["neural_first", "ibm1_first"],
    )
    parser.add_argument("--candidate_topks", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--neural_score_modes",
        nargs="+",
        choices=["raw", "reciprocal_rank", "linear_rank"],
        default=["raw", "reciprocal_rank"],
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=["none", "minmax", "zscore", "rank"],
        default=["minmax", "rank"],
    )
    parser.add_argument(
        "--formulas",
        nargs="+",
        choices=["weighted_sum", "weighted_product"],
        default=["weighted_sum", "weighted_product"],
    )
    parser.add_argument("--length_penalties", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument(
        "--weights",
        nargs="+",
        default=[
            "1.0,0.0,0.0",
            "0.75,0.25,0.0",
            "0.5,0.5,0.0",
            "0.25,0.75,0.0",
            "0.0,1.0,0.0",
            "0.5,0.4,0.1",
        ],
        help="Each value is alpha,beta,gamma.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prune_stale", action="store_true")

    parser.add_argument(
        "--candidate_generator",
        choices=["neural_first", "ibm1_first"],
        default="neural_first",
    )
    parser.add_argument("--candidate_topk", type=int, default=10)
    parser.add_argument(
        "--neural_score_mode",
        choices=["raw", "reciprocal_rank", "linear_rank"],
        default="reciprocal_rank",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "minmax", "zscore", "rank"],
        default="minmax",
    )
    parser.add_argument(
        "--formula",
        choices=["weighted_sum", "weighted_product"],
        default="weighted_sum",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    main(parser.parse_args())
