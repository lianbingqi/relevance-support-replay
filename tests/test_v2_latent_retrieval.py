import numpy as np

from src.representation.similarity import (
    cosine_similarity_matrix,
    topk_indices,
    topk_label_precision,
)


def test_topk_cosine_retrieval():
    bank = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        ["A", "A", "B", "B"]
    )

    query = np.asarray(
        [
            [1.0, 0.05],
        ],
        dtype=np.float32,
    )

    sim = cosine_similarity_matrix(
        query,
        bank,
    )

    idx = topk_indices(
        sim,
        2,
    )

    assert set(
        idx[0].tolist()
    ) == {0, 1}

    result = topk_label_precision(
        query,
        bank,
        labels,
        target_label="A",
        k=2,
    )

    assert np.isclose(
        result[
            "precision_per_query"
        ][0],
        1.0,
    )

    assert (
        result[
            "same_vs_other_margin"
        ][0]
        > 0
    )
