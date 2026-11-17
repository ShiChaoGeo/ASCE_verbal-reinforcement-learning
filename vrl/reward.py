"""Scalar reward."""

from __future__ import annotations

from vrl.environment import ACTION_SPACE


def scalar_reward(y_pred: list[str], y_true: list[str]) -> dict:
    """Accuracy over one sounding's segments."""
    if len(y_pred) != len(y_true):
        raise ValueError(f"Length mismatch: pred={len(y_pred)} true={len(y_true)}")

    n = len(y_true)
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    per_class = {}
    for label in ACTION_SPACE:
        idx = [i for i, t in enumerate(y_true) if t == label]
        if idx:
            per_class[label] = sum(y_pred[i] == label for i in idx) / len(idx)

    return {
        "n_segments": n,
        "n_correct": correct,
        "accuracy": correct / n if n else float("nan"),
        "per_class_recall": per_class,
    }


