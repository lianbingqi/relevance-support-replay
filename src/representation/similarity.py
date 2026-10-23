from __future__ import annotations

import numpy as np


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError("x must be 2D")

    denom = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    denom = np.maximum(
        denom,
        1e-8,
    )

    return (
        x / denom
    ).astype(np.float32)


def cosine_similarity_matrix(
    query: np.ndarray,
    bank: np.ndarray,
) -> np.ndarray:
    q = l2_normalize_rows(
        query
    )

    b = l2_normalize_rows(
        bank
    )

    return (
        q @ b.T
    ).astype(np.float32)


def topk_indices(
    similarity: np.ndarray,
    k: int,
) -> np.ndarray:
    similarity = np.asarray(
        similarity
    )

    if similarity.ndim != 2:
        raise ValueError(
            "similarity must be 2D"
        )

    if k <= 0:
        raise ValueError(
            "k must be positive"
        )

    if k > similarity.shape[1]:
        raise ValueError(
            "k exceeds bank size"
        )

    idx = np.argpartition(
        -similarity,
        kth=int(k) - 1,
        axis=1,
    )[:, :int(k)]

    row = np.arange(
        similarity.shape[0]
    )[:, None]

    order = np.argsort(
        -similarity[
            row,
            idx,
        ],
        axis=1,
    )

    return idx[
        row,
        order,
    ]


def topk_label_precision(
    query_z: np.ndarray,
    bank_z: np.ndarray,
    bank_labels: np.ndarray,
    target_label: str,
    k: int,
):
    similarity = cosine_similarity_matrix(
        query_z,
        bank_z,
    )

    idx = topk_indices(
        similarity,
        k,
    )

    labels = np.asarray(
        bank_labels
    )[idx]

    precision_per_query = np.mean(
        labels == target_label,
        axis=1,
    ).astype(np.float32)

    row = np.arange(
        len(
            query_z
        )
    )

    best_similarity = similarity[
        row,
        idx[:, 0],
    ]

    target_mask = (
        np.asarray(
            bank_labels
        )
        == target_label
    )

    other_mask = ~target_mask

    best_target = np.max(
        similarity[
            :,
            target_mask
        ],
        axis=1,
    )

    best_other = np.max(
        similarity[
            :,
            other_mask
        ],
        axis=1,
    )

    margin = (
        best_target
        - best_other
    ).astype(
        np.float32
    )

    return {
        "indices": idx,
        "precision_per_query": precision_per_query,
        "best_similarity": best_similarity,
        "same_vs_other_margin": margin,
    }
