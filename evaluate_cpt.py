"""Two accuracies for one CPT sounding, measured against two different references."""

from __future__ import annotations

import sys

from cpt_segmentation import load_cpt, segment_ic
from ground_truth import (
    CELLS_PER_M,
    aligned_stratification,
    column_for,
    depth_agreement,
)
from stratification_report import engineer_labels
from vrl.environment import cpt_path
from vrl.reward import scalar_reward
from vrl.trajectory import Trajectory


def metric_a(cpt_id: int = 1, pen: float = 5.0) -> dict:
    """Depth-wise agreement with the engineer cross-section, for one sounding."""
    segments = segment_ic(load_cpt(cpt_path(cpt_id)), pen=pen)
    interpreted = engineer_labels(cpt_id)
    if len(interpreted) != len(segments):
        raise ValueError(
            f"CPT{cpt_id}: {len(segments)} segments but {len(interpreted)} engineer labels."
        )
    aligned = aligned_stratification(cpt_id, segments, interpreted)
    result = depth_agreement(aligned, boundary_tolerance_m=1 / CELLS_PER_M)
    result["cpt_id"] = cpt_id
    result["reference"] = f"ground_truth column {column_for(cpt_id)}"
    return result


def metric_b(cpt_id: int = 1, policy_version: str = "v0") -> dict | None:
    """Segment accuracy of a locked policy action against the engineer's labels."""
    trajectory = Trajectory(cpt_id=cpt_id, policy_version=policy_version)
    if not trajectory.is_locked:
        return None

    y_pred = [a["predicted_label"] for a in trajectory.load_action()]
    result = scalar_reward(y_pred, engineer_labels(cpt_id))
    result.update({"cpt_id": cpt_id, "policy_version": policy_version,
                   "reference": "engineer_label.csv"})
    return result


def _report(cpt_id: int, policy_version: str) -> None:
    a = metric_a(cpt_id)
    tol = a["boundary_tolerance_m"]

    print(f"CPT{cpt_id}")
    print("METRIC A — stratification vs engineer cross-section (depth-weighted)")
    print(f"  reference              : {a['reference']}")
    print(f"  profile length         : {a['profile_length_m']:.3f} m")
    print(f"  agreement              : {a['agreement']:.4f} "
          f"({a['disagreement_m']:.3f} m disagree)")
    print(f"  allowing {tol:.3f} m boundary error : "
          f"{a['agreement_boundary_tolerant']:.4f}")
    print(f"    at layer boundaries  : {a['disagreement_at_boundary_m']:.3f} m "
          f"(placement uncertainty)")
    print(f"    interior             : {a['disagreement_interior_m']:.3f} m "
          f"(different lithology)")
    print("  per-class recall (thickness-weighted):")
    for name, value in a["per_class_recall"].items():
        print(f"    {name:26s} {value:.4f}")

    print("\nMETRIC B — policy action vs engineer segment calls (shared segmentation)")
    b = metric_b(cpt_id, policy_version)
    if b is None:
        print(f"  no locked action for CPT{cpt_id} under {policy_version}.")
        print("  run a search round first: python main.py search-round")
    else:
        print(f"  reference              : {b['reference']}")
        print(f"  accuracy               : {b['accuracy']:.4f} "
              f"({b['n_correct']}/{b['n_segments']} segments)")
        print("  per-class recall:")
        for name, value in b["per_class_recall"].items():
            print(f"    {name:26s} {value:.4f}")


if __name__ == "__main__":
    cpt_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    policy_version = sys.argv[2] if len(sys.argv) > 2 else "v0"
    _report(cpt_id, policy_version)
