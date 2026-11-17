"""Engineer soil labels of the PELT segments, and their depth agreement with the section."""

from __future__ import annotations

import csv
from pathlib import Path

from cpt_segmentation import load_cpt, segment_ic
from ground_truth import (
    CELLS_PER_M,
    LABEL_TO_ID,
    aligned_stratification,
    column_for,
    depth_agreement,
)

DATA_DIR = Path(__file__).parent

ENGINEER_LABELS = DATA_DIR / "engineer_label.csv"

SOIL_NAME_TO_ID = LABEL_TO_ID


def engineer_labels(cpt_id: int) -> list[str]:
    """The engineer's segment labels for one sounding, in depth order."""
    with ENGINEER_LABELS.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["cpt_id"]) == cpt_id]
    if not rows:
        raise KeyError(f"No engineer labels for CPT {cpt_id} in {ENGINEER_LABELS.name}")

    labels = [r["label"] for r in sorted(rows, key=lambda r: int(r["segment"]))]
    unknown = sorted(set(labels) - set(SOIL_NAME_TO_ID))
    if unknown:
        raise ValueError(f"Unknown engineer label(s) in {ENGINEER_LABELS.name}: {unknown}")
    return labels


if __name__ == "__main__":
    cpt_id = 1
    labels = engineer_labels(cpt_id)
    df = load_cpt()
    segments = segment_ic(df, pen=5.0)

    print(f"Labels: {ENGINEER_LABELS.name} (CPT{cpt_id})")
    print(f"Segments labelled: {len(segments)}")
    print("\nSegment → soil_id / label")
    for i, (seg, name) in enumerate(zip(segments, labels), start=1):
        z0 = -float(seg["depth"].iloc[0])
        z1 = -float(seg["depth"].iloc[-1])
        print(
            f"  [{i:02d}] {z0:.3f}-{z1:.3f} m  "
            f"id={SOIL_NAME_TO_ID[name]}  {name}  (n={len(seg)})"
        )

    aligned = aligned_stratification(cpt_id, segments, labels)
    agree = depth_agreement(aligned, boundary_tolerance_m=1 / CELLS_PER_M)

    print(f"\nReference: ground_truth column {column_for(cpt_id)}")
    print(f"Profile length: {agree['profile_length_m']:.2f} m")
    print(f"Stratification agreement: {agree['agreement']:.4f} "
          f"({agree['disagreement_m']:.2f} m disagree)")
    print(f"  within {agree['boundary_tolerance_m']:.2f} m of a reference boundary: "
          f"{agree['agreement_boundary_tolerant']:.4f} "
          f"({agree['disagreement_interior_m']:.2f} m interior disagreement)")
    print("  per-class recall (thickness-weighted):")
    for name, value in agree["per_class_recall"].items():
        print(f"    {name:26s} {value:.4f}")
