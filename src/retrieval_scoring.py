# #!/usr/bin/env python3
# """Shared retrieval scoring for bilingual sentence retrieval.

# The functions in this module operate on the complete source-target similarity
# matrix. This is important for CSLS and Artetxe-Schwenk ratio-margin scoring:
# nearest-neighbour statistics must be computed over the full evaluation pool,
# not over an already truncated candidate list.
# """

# from __future__ import annotations

# from typing import Tuple

# import numpy as np

# VALID_RETRIEVALS = ("cosine", "csls", "margin_ratio")


# def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
#     """Return row-wise L2-normalized float32 embeddings."""
#     array = np.asarray(matrix, dtype=np.float32)
#     if array.ndim != 2:
#         raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
#     norms = np.linalg.norm(array, axis=1, keepdims=True)
#     norms[norms == 0.0] = 1.0
#     return (array / norms).astype(np.float32, copy=False)


# def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
#     """Return score-sorted top-k candidate indices for every query."""
#     if scores.ndim != 2:
#         raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
#     if scores.shape[1] == 0:
#         raise ValueError("The candidate pool is empty")

#     k = int(max(1, min(int(topk), scores.shape[1])))
#     if k == scores.shape[1]:
#         return np.argsort(-scores, axis=1, kind="stable").astype(np.int64)

#     unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
#     selected_scores = np.take_along_axis(scores, unsorted, axis=1)
#     order = np.argsort(-selected_scores, axis=1, kind="stable")
#     return np.take_along_axis(unsorted, order, axis=1).astype(np.int64)


# def cosine_similarity_matrix(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
# ) -> np.ndarray:
#     """Compute the complete cosine-similarity matrix."""
#     query = l2_normalize_rows(query_embeddings)
#     candidate = l2_normalize_rows(candidate_embeddings)
#     if query.shape[1] != candidate.shape[1]:
#         raise ValueError(
#             "Query and candidate embedding dimensions differ: "
#             f"{query.shape[1]} vs {candidate.shape[1]}"
#         )
#     return (query @ candidate.T).astype(np.float32, copy=False)


# def neighborhood_means(
#     cosine_scores: np.ndarray,
#     neighborhood_k: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute mean top-k similarities on the query and candidate sides.

#     The query-side value is the average of each query's top-k candidate
#     similarities. The candidate-side value is the average of each candidate's
#     top-k query similarities. Both are computed from the complete matrix.
#     """
#     scores = np.asarray(cosine_scores, dtype=np.float32)
#     if scores.ndim != 2:
#         raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
#     if scores.shape[0] == 0 or scores.shape[1] == 0:
#         raise ValueError("The score matrix must be non-empty")
#     if neighborhood_k < 1:
#         raise ValueError("neighborhood_k must be positive")

#     query_k = min(int(neighborhood_k), scores.shape[1])
#     candidate_k = min(int(neighborhood_k), scores.shape[0])

#     query_partition = np.partition(scores, scores.shape[1] - query_k, axis=1)
#     query_means = query_partition[:, -query_k:].mean(axis=1, dtype=np.float64)

#     candidate_partition = np.partition(scores, scores.shape[0] - candidate_k, axis=0)
#     candidate_means = candidate_partition[-candidate_k:, :].mean(
#         axis=0, dtype=np.float64
#     )

#     return (
#         query_means.astype(np.float32),
#         candidate_means.astype(np.float32),
#     )


# def retrieval_score_matrix(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
#     retrieval: str,
#     neighborhood_k: int = 10,
#     epsilon: float = 1e-8,
# ) -> np.ndarray:
#     """Compute cosine, CSLS, or ratio-margin scores.

#     Ratio margin follows Artetxe and Schwenk (2019): the pairwise cosine score
#     is divided by the average source- and target-side neighbourhood similarity.
#     A small signed epsilon is used only when that denominator is numerically
#     zero.
#     """
#     if retrieval not in VALID_RETRIEVALS:
#         raise ValueError(
#             f"Unknown retrieval criterion {retrieval!r}; "
#             f"choose one of {VALID_RETRIEVALS}"
#         )
#     if neighborhood_k < 1:
#         raise ValueError("neighborhood_k must be positive")

#     cosine = cosine_similarity_matrix(query_embeddings, candidate_embeddings)
#     if retrieval == "cosine":
#         return cosine

#     query_means, candidate_means = neighborhood_means(cosine, neighborhood_k)
#     if retrieval == "csls":
#         return (
#             2.0 * cosine
#             - query_means[:, None]
#             - candidate_means[None, :]
#         ).astype(np.float32)

#     neighborhood_average = (
#         query_means[:, None] + candidate_means[None, :]
#     ) / 2.0
#     safe_denominator = np.where(
#         np.abs(neighborhood_average) < epsilon,
#         np.where(neighborhood_average < 0.0, -epsilon, epsilon),
#         neighborhood_average,
#     )
#     return (cosine / safe_denominator).astype(np.float32)


# def retrieve_topk(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
#     retrieval: str,
#     topk: int,
#     neighborhood_k: int = 10,
#     epsilon: float = 1e-8,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute full-pool scores and return top-k indices plus the score matrix."""
#     scores = retrieval_score_matrix(
#         query_embeddings=query_embeddings,
#         candidate_embeddings=candidate_embeddings,
#         retrieval=retrieval,
#         neighborhood_k=neighborhood_k,
#         epsilon=epsilon,
#     )
#     return topk_from_scores(scores, topk), scores

















# #!/usr/bin/env python3
# """Shared retrieval scoring for bilingual sentence retrieval.

# The functions in this module operate on the complete source-target similarity
# matrix. This is important for CSLS and Artetxe-Schwenk ratio-margin scoring:
# nearest-neighbour statistics must be computed over the full evaluation pool,
# not over an already truncated candidate list.
# """

# from __future__ import annotations

# from typing import Tuple

# import numpy as np

# VALID_RETRIEVALS = ("cosine", "csls", "margin_ratio")


# def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
#     """Return row-wise L2-normalized float32 embeddings."""
#     array = np.asarray(matrix, dtype=np.float32)
#     if array.ndim != 2:
#         raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
#     norms = np.linalg.norm(array, axis=1, keepdims=True)
#     norms[norms == 0.0] = 1.0
#     return (array / norms).astype(np.float32, copy=False)


# def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
#     """Return score-sorted top-k candidate indices for every query."""
#     if scores.ndim != 2:
#         raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
#     if scores.shape[1] == 0:
#         raise ValueError("The candidate pool is empty")

#     k = int(max(1, min(int(topk), scores.shape[1])))
#     if k == scores.shape[1]:
#         return np.argsort(-scores, axis=1, kind="stable").astype(np.int64)

#     unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
#     selected_scores = np.take_along_axis(scores, unsorted, axis=1)
#     order = np.argsort(-selected_scores, axis=1, kind="stable")
#     return np.take_along_axis(unsorted, order, axis=1).astype(np.int64)


# def cosine_similarity_matrix(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
# ) -> np.ndarray:
#     """Compute the complete cosine-similarity matrix."""
#     query = l2_normalize_rows(query_embeddings)
#     candidate = l2_normalize_rows(candidate_embeddings)
#     if query.shape[1] != candidate.shape[1]:
#         raise ValueError(
#             "Query and candidate embedding dimensions differ: "
#             f"{query.shape[1]} vs {candidate.shape[1]}"
#         )
#     return (query @ candidate.T).astype(np.float32, copy=False)


# def neighborhood_means(
#     cosine_scores: np.ndarray,
#     neighborhood_k: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute mean top-k similarities on the query and candidate sides.

#     The query-side value is the average of each query's top-k candidate
#     similarities. The candidate-side value is the average of each candidate's
#     top-k query similarities. Both are computed from the complete matrix.
#     """
#     scores = np.asarray(cosine_scores, dtype=np.float32)
#     if scores.ndim != 2:
#         raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
#     if scores.shape[0] == 0 or scores.shape[1] == 0:
#         raise ValueError("The score matrix must be non-empty")
#     if neighborhood_k < 1:
#         raise ValueError("neighborhood_k must be positive")

#     query_k = min(int(neighborhood_k), scores.shape[1])
#     candidate_k = min(int(neighborhood_k), scores.shape[0])

#     query_partition = np.partition(scores, scores.shape[1] - query_k, axis=1)
#     query_means = query_partition[:, -query_k:].mean(axis=1, dtype=np.float64)

#     candidate_partition = np.partition(scores, scores.shape[0] - candidate_k, axis=0)
#     candidate_means = candidate_partition[-candidate_k:, :].mean(
#         axis=0, dtype=np.float64
#     )

#     return (
#         query_means.astype(np.float32),
#         candidate_means.astype(np.float32),
#     )


# def retrieval_score_matrix(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
#     retrieval: str,
#     neighborhood_k: int = 10,
#     epsilon: float = 1e-8,
# ) -> np.ndarray:
#     """Compute cosine, CSLS, or ratio-margin scores.

#     Ratio margin follows Artetxe and Schwenk (2019): the pairwise cosine score
#     is divided by the average source- and target-side neighbourhood similarity.
#     The denominator is lower-bounded by ``epsilon`` for numerical stability.
#     """
#     if retrieval not in VALID_RETRIEVALS:
#         raise ValueError(
#             f"Unknown retrieval criterion {retrieval!r}; "
#             f"choose one of {VALID_RETRIEVALS}"
#         )
#     if neighborhood_k < 1:
#         raise ValueError("neighborhood_k must be positive")

#     cosine = cosine_similarity_matrix(query_embeddings, candidate_embeddings)
#     if retrieval == "cosine":
#         return cosine

#     query_means, candidate_means = neighborhood_means(cosine, neighborhood_k)
#     if retrieval == "csls":
#         return (
#             2.0 * cosine
#             - query_means[:, None]
#             - candidate_means[None, :]
#         ).astype(np.float32)

#     neighborhood_average = (
#         query_means[:, None] + candidate_means[None, :]
#     ) / 2.0
#     safe_denominator = np.maximum(neighborhood_average, epsilon)
#     return (cosine / safe_denominator).astype(np.float32)


# def score_retrieval(
#     source_embeddings: np.ndarray,
#     target_embeddings: np.ndarray,
#     criterion: str,
#     neighborhood_k: int = 10,
#     epsilon: float = 1e-8,
# ) -> np.ndarray:
#     """Shared reviewer-facing interface for all three retrieval criteria."""
#     return retrieval_score_matrix(
#         query_embeddings=source_embeddings,
#         candidate_embeddings=target_embeddings,
#         retrieval=criterion,
#         neighborhood_k=neighborhood_k,
#         epsilon=epsilon,
#     )


# def retrieve_topk(
#     query_embeddings: np.ndarray,
#     candidate_embeddings: np.ndarray,
#     retrieval: str,
#     topk: int,
#     neighborhood_k: int = 10,
#     epsilon: float = 1e-8,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute full-pool scores and return top-k indices plus the score matrix."""
#     scores = retrieval_score_matrix(
#         query_embeddings=query_embeddings,
#         candidate_embeddings=candidate_embeddings,
#         retrieval=retrieval,
#         neighborhood_k=neighborhood_k,
#         epsilon=epsilon,
#     )
#     return topk_from_scores(scores, topk), scores

















#!/usr/bin/env python3
"""Shared retrieval scoring for bilingual sentence retrieval.

The functions in this module operate on the complete source-target similarity
matrix. This is important for CSLS and Artetxe-Schwenk ratio-margin scoring:
nearest-neighbour statistics must be computed over the full evaluation pool,
not over an already truncated candidate list.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

VALID_RETRIEVALS = ("cosine", "csls", "margin_ratio")


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 embeddings."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(np.float32, copy=False)


def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return score-sorted top-k candidate indices for every query."""
    score_array = np.asarray(scores)
    if score_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D score matrix, got shape {score_array.shape}"
        )
    if score_array.shape[1] == 0:
        raise ValueError("The candidate pool is empty")

    k = int(max(1, min(int(topk), score_array.shape[1])))
    if k == score_array.shape[1]:
        return np.argsort(
            -score_array,
            axis=1,
            kind="stable",
        ).astype(np.int64)

    unsorted = np.argpartition(
        -score_array,
        kth=k - 1,
        axis=1,
    )[:, :k]
    selected_scores = np.take_along_axis(
        score_array,
        unsorted,
        axis=1,
    )
    order = np.argsort(
        -selected_scores,
        axis=1,
        kind="stable",
    )
    return np.take_along_axis(
        unsorted,
        order,
        axis=1,
    ).astype(np.int64)


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute the complete cosine-similarity matrix."""
    query = l2_normalize_rows(query_embeddings)
    candidate = l2_normalize_rows(candidate_embeddings)

    if query.shape[1] != candidate.shape[1]:
        raise ValueError(
            "Query and candidate embedding dimensions differ: "
            f"{query.shape[1]} vs {candidate.shape[1]}"
        )

    return (query @ candidate.T).astype(np.float32, copy=False)


def neighborhood_means(
    cosine_scores: np.ndarray,
    neighborhood_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute query- and candidate-side top-k neighborhood means."""
    scores = np.asarray(cosine_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("The score matrix must be non-empty")
    if neighborhood_k < 1:
        raise ValueError("neighborhood_k must be positive")

    query_k = min(int(neighborhood_k), scores.shape[1])
    candidate_k = min(int(neighborhood_k), scores.shape[0])

    query_partition = np.partition(
        scores,
        scores.shape[1] - query_k,
        axis=1,
    )
    query_means = query_partition[:, -query_k:].mean(
        axis=1,
        dtype=np.float64,
    )

    candidate_partition = np.partition(
        scores,
        scores.shape[0] - candidate_k,
        axis=0,
    )
    candidate_means = candidate_partition[-candidate_k:, :].mean(
        axis=0,
        dtype=np.float64,
    )

    return (
        query_means.astype(np.float32),
        candidate_means.astype(np.float32),
    )


def retrieval_score_matrix(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    retrieval: str,
    neighborhood_k: int = 10,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute cosine, CSLS, or Artetxe-Schwenk ratio-margin scores."""
    if retrieval not in VALID_RETRIEVALS:
        raise ValueError(
            f"Unknown retrieval criterion {retrieval!r}; "
            f"choose one of {VALID_RETRIEVALS}"
        )
    if neighborhood_k < 1:
        raise ValueError("neighborhood_k must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    cosine = cosine_similarity_matrix(
        query_embeddings,
        candidate_embeddings,
    )
    if retrieval == "cosine":
        return cosine

    query_means, candidate_means = neighborhood_means(
        cosine,
        neighborhood_k,
    )

    if retrieval == "csls":
        return (
            2.0 * cosine
            - query_means[:, None]
            - candidate_means[None, :]
        ).astype(np.float32)

    neighborhood_average = (
        query_means[:, None] + candidate_means[None, :]
    ) / 2.0
    safe_denominator = np.maximum(
        neighborhood_average,
        epsilon,
    )
    return (cosine / safe_denominator).astype(np.float32)


def score_retrieval(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    criterion: str,
    neighborhood_k: int = 10,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return a complete score matrix for the requested retrieval criterion."""
    return retrieval_score_matrix(
        query_embeddings=source_embeddings,
        candidate_embeddings=target_embeddings,
        retrieval=criterion,
        neighborhood_k=neighborhood_k,
        epsilon=epsilon,
    )


def retrieve_topk(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    retrieval: str,
    topk: int,
    neighborhood_k: int = 10,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return score-sorted top-k indices and the complete score matrix."""
    scores = retrieval_score_matrix(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        retrieval=retrieval,
        neighborhood_k=neighborhood_k,
        epsilon=epsilon,
    )
    return topk_from_scores(scores, topk), scores
