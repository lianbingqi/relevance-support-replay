import numpy as np

from scripts.run_frozen_trajectory_control_v2 import (
    array_hash,
    balanced_idx,
    bundle_hash,
)


def test_bundle_hash_is_order_stable_and_sensitive():
    a = {
        "x": np.asarray(
            [[1.0, 2.0]],
            dtype=np.float32,
        ),
        "y": np.asarray(
            [3],
            dtype=np.int64,
        ),
    }

    b = {
        "y": np.asarray(
            [3],
            dtype=np.int64,
        ),
        "x": np.asarray(
            [[1.0, 2.0]],
            dtype=np.float32,
        ),
    }

    assert bundle_hash(a) == bundle_hash(b)

    c = {
        **a,
        "y": np.asarray(
            [4],
            dtype=np.int64,
        ),
    }

    assert bundle_hash(a) != bundle_hash(c)


def test_balanced_idx_is_balanced_and_reproducible():
    labels = np.asarray(
        ["A"] * 10 + ["B"] * 10
    )

    idx1 = balanced_idx(
        labels,
        max_per_context=4,
        seed=1,
    )

    idx2 = balanced_idx(
        labels,
        max_per_context=4,
        seed=1,
    )

    assert np.array_equal(
        idx1,
        idx2,
    )

    kept = labels[idx1]

    assert len(idx1) == 8
    assert int(np.sum(kept == "A")) == 4
    assert int(np.sum(kept == "B")) == 4


def test_array_hash_is_sensitive():
    x = np.asarray(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    y = x.copy()
    y[0, 0] += 1.0

    assert array_hash(x) != array_hash(y)
