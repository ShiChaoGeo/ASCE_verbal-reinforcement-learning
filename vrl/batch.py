"""Batch interpretation, scoring and reflection over the training soundings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from stratification_report import engineer_labels as reference_labels
from vrl.environment import observe
from vrl.environment import parse_depth_label
from vrl.policy import Policy, act
from vrl.providers import DEFAULT_PROVIDER
from vrl.reflect import self_reflect
from vrl.reward import scalar_reward
from vrl.trajectory import Trajectory


@dataclass
class BatchScore:
    """Pooled Metric B over the whole batch, plus the per-sounding breakdown."""

    per_cpt: dict[int, dict]
    n_segments: int
    n_correct: int
    accuracy: float
    per_class_recall: dict[str, float]
    confusions: dict[str, int] = field(default_factory=dict)
    per_segment: dict[int, list[dict]] = field(default_factory=dict)
    metric_a: float | None = None
    metric_a_per_cpt: dict[int, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "per_class_recall": self.per_class_recall,
            "confusions": self.confusions,
            "per_cpt": {str(k): v for k, v in self.per_cpt.items()},
            "per_segment": {str(k): v for k, v in self.per_segment.items()},
            "metric_a": self.metric_a,
            "metric_a_per_cpt": {str(k): v for k, v in self.metric_a_per_cpt.items()},
        }


def _metric_a(cpt_id: int, predicted: list[str], pen: float) -> dict | None:
    """Metric A for a locked action: depth-weighted agreement with the section."""
    try:
        from cpt_segmentation import load_cpt, segment_ic
        from vrl.environment import cpt_path
        from ground_truth import CELLS_PER_M, aligned_stratification, depth_agreement

        segments = segment_ic(load_cpt(cpt_path(cpt_id)), pen=pen)
        if len(segments) != len(predicted):
            return None
        aligned = aligned_stratification(cpt_id, segments, predicted)
        return depth_agreement(aligned, boundary_tolerance_m=1 / CELLS_PER_M)
    except Exception:
        return None


def _episode_pen(trajectory: Trajectory, default: float = 5.0) -> float:
    meta = trajectory.path("episode_meta.json")
    if meta.is_file():
        try:
            return float(json.loads(meta.read_text(encoding="utf-8")).get("pen", default))
        except (ValueError, json.JSONDecodeError):
            pass
    return default


def score_batch(cpt_ids: list[int], policy_version: str) -> BatchScore:
    """Pool Metric B across the batch from the locked actions already on disk."""
    per_cpt: dict[int, dict] = {}
    per_segment: dict[int, list[dict]] = {}
    metric_a_per_cpt: dict[int, dict] = {}
    y_pred_all: list[str] = []
    y_true_all: list[str] = []

    for cpt_id in cpt_ids:
        trajectory = Trajectory(cpt_id=cpt_id, policy_version=policy_version)
        action = trajectory.load_action()
        y_pred = [a["predicted_label"] for a in action]
        y_true = reference_labels(cpt_id)
        if len(y_pred) != len(y_true):
            raise ValueError(
                f"CPT{cpt_id}: {len(y_pred)} predictions but {len(y_true)} "
                "reference labels. engineer_label.csv is tied to the PELT "
                "penalty the action was produced with."
            )
        result = scalar_reward(y_pred, y_true)
        result["cpt_id"] = cpt_id
        per_cpt[cpt_id] = result

        try:
            depths = {int(c["segment"]): c.get("depth_label", "")
                      for c in trajectory.load_state()}
        except FileNotFoundError:
            depths = {}
        rows = []
        for item, truth in zip(action, y_true):
            seg = int(item["segment"])
            top, base = parse_depth_label(depths.get(seg, ""))
            rows.append({
                "segment": seg,
                "depth_label": depths.get(seg, ""),
                "depth_top_m": top,
                "depth_base_m": base,
                "predicted_label": item["predicted_label"],
                "engineer_label": truth,
                "correct": item["predicted_label"] == truth,
            })
        per_segment[cpt_id] = rows

        agreement = _metric_a(cpt_id, y_pred, _episode_pen(trajectory))
        if agreement is not None:
            metric_a_per_cpt[cpt_id] = agreement

        y_pred_all += y_pred
        y_true_all += y_true

    pooled = scalar_reward(y_pred_all, y_true_all)

    confusions: dict[str, int] = {}
    for pred, true in zip(y_pred_all, y_true_all):
        if pred != true:
            confusions[f"{true} -> {pred}"] = confusions.get(f"{true} -> {pred}", 0) + 1

    metric_a = None
    if metric_a_per_cpt:
        total = sum(r["profile_length_m"] for r in metric_a_per_cpt.values())
        if total > 0:
            metric_a = sum(r["agreement"] * r["profile_length_m"]
                           for r in metric_a_per_cpt.values()) / total

    return BatchScore(
        metric_a=metric_a,
        metric_a_per_cpt=metric_a_per_cpt,
        per_cpt=per_cpt,
        n_segments=pooled["n_segments"],
        n_correct=pooled["n_correct"],
        accuracy=pooled["accuracy"],
        per_class_recall=pooled["per_class_recall"],
        confusions=dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
        per_segment=per_segment,
    )


def interpret_batch(
    cpt_ids: list[int],
    policy_version: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    pen: float = 5.0,
) -> list[Trajectory]:
    """Run one policy version over every sounding in the batch and lock each."""
    trajectories = []
    for cpt_id in cpt_ids:
        trajectory = Trajectory(cpt_id=cpt_id, policy_version=policy_version)
        if trajectory.is_locked:
            print(f"  CPT{cpt_id}: already locked under {policy_version}, reusing")
        else:
            state = observe(cpt_id, pen=pen)
            trajectory.save_state(state.cards)
            action = act(Policy.load(policy_version), state,
                         provider=provider, model=model)
            trajectory.save_action(action, n_segments=state.n_segments)
            trajectory.save_meta({
                "provider": provider,
                "model": model,
                "pen": pen,
                "mode": "batch_training",
                "batch": [int(c) for c in cpt_ids],
            })
            print(f"  CPT{cpt_id}: {len(action)} segments interpreted and locked")
        trajectories.append(trajectory)
    return trajectories


def reflect_batch(
    trajectories: list[Trajectory],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> list[dict]:
    """Two-stage self-reflection on each sounding of the batch."""
    payloads = []
    for trajectory in trajectories:
        if trajectory.has_verbal_reward:
            print(f"  CPT{trajectory.cpt_id}: verbal reward exists, reusing")
            payload = trajectory.load_verbal_reward()
        else:
            payload = self_reflect(trajectory, provider=provider, model=model)
        payloads.append(payload)
    return payloads
