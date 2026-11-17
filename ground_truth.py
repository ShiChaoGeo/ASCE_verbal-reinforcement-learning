"""Engineer cross-section ground truth, resampled to CPT segmentation resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
GRID_NPY = DATA_DIR / "ground_truth.npy"
GRID_XLSX = DATA_DIR / "segmentation2.xlsx"

CELLS_PER_M = 6
PROFILE_DEPTH_M = 40.0

ID_TO_LABEL = {
    0: "alluvial_sand",
    1: "alluvial_clay",
    2: "marine_deposit",
    3: "disturbed_marine_deposit",
    4: "fill",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


CPT_TO_COLUMN = {1: 11, 2: 24, 3: 44, 4: 54}


def load_grid() -> np.ndarray:
    """The cross-section at ``CELLS_PER_M``, upsampled from the source if needed."""
    if GRID_NPY.is_file():
        grid = np.load(GRID_NPY)
    elif GRID_XLSX.is_file():
        grid = pd.read_excel(GRID_XLSX, header=None).to_numpy()
    else:
        raise FileNotFoundError(f"Neither {GRID_NPY.name} nor {GRID_XLSX.name} found.")

    want = int(round(CELLS_PER_M * PROFILE_DEPTH_M))
    have = grid.shape[0]
    if have == want:
        return grid
    if want % have:
        raise ValueError(
            f"{GRID_NPY.name} has {have} rows but CELLS_PER_M={CELLS_PER_M} "
            f"expects {want}, which is not a whole multiple. Re-export the "
            f"cross-section, or set CELLS_PER_M to a multiple of {have / PROFILE_DEPTH_M:g}."
        )
    return np.repeat(grid, want // have, axis=0)


def column_for(cpt_id: int) -> int:
    if cpt_id not in CPT_TO_COLUMN:
        raise KeyError(
            f"No ground-truth column known for CPT {cpt_id}. "
            f"Known: {sorted(CPT_TO_COLUMN)}"
        )
    return CPT_TO_COLUMN[cpt_id]


def profile(cpt_id: int, grid: np.ndarray | None = None) -> np.ndarray:
    """The label column for one sounding at ``CELLS_PER_M``, shallow to deep."""
    grid = load_grid() if grid is None else grid
    return grid[:, column_for(cpt_id)]


def upsample_to_depths(cpt_id: int, depths: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Label every CPT sample depth by the cross-section cell containing it."""
    col = profile(cpt_id, grid)
    z = np.abs(np.asarray(depths, dtype=float))
    idx = np.clip((z * CELLS_PER_M).astype(int), 0, len(col) - 1)
    return col[idx]


def segment_labels(
    cpt_id: int,
    segments: list[pd.DataFrame],
    grid: np.ndarray | None = None,
    as_names: bool = True,
) -> list[str | int]:
    """One ground-truth label per PELT segment, by majority vote over its depth span."""
    col = profile(cpt_id, grid)
    out: list[str | int] = []
    for seg in segments:
        z0 = abs(float(seg["depth"].iloc[0]))
        z1 = abs(float(seg["depth"].iloc[-1]))
        lo, hi = sorted((z0, z1))
        k0 = int(np.floor(lo * CELLS_PER_M))
        k1 = int(np.ceil(hi * CELLS_PER_M))
        k0 = max(0, min(k0, len(col) - 1))
        k1 = max(k0 + 1, min(k1, len(col)))
        cells = col[k0:k1]
        values, counts = np.unique(cells, return_counts=True)
        winner = int(values[counts.argmax()])
        out.append(ID_TO_LABEL[winner] if as_names else winner)
    return out


def aligned_stratification(
    cpt_id: int,
    segments: list[pd.DataFrame],
    interpreted_labels: list[str],
    grid: np.ndarray | None = None,
) -> dict:
    """Put the ground truth and an interpretation on one shared depth axis."""
    if len(interpreted_labels) != len(segments):
        raise ValueError(
            f"Label count ({len(interpreted_labels)}) != segment count ({len(segments)})."
        )

    depth = np.concatenate([s["depth"].to_numpy(dtype=float) for s in segments])
    gt_point_ids = upsample_to_depths(cpt_id, depth, grid)

    interp_point_ids = np.concatenate(
        [np.full(len(s), LABEL_TO_ID[lab], dtype=int)
         for s, lab in zip(segments, interpreted_labels)]
    )

    if len(gt_point_ids) != len(interp_point_ids):
        raise AssertionError("Depth axes disagree after alignment.")

    return {
        "cpt_id": cpt_id,
        "depth": depth,
        "ground_truth_ids": gt_point_ids,
        "interpreted_ids": interp_point_ids,
        "ground_truth_segments": segment_labels(cpt_id, segments, grid),
        "interpreted_segments": list(interpreted_labels),
    }


def _sample_thickness(depth: np.ndarray) -> np.ndarray:
    """Depth interval each sample represents, from midpoints between samples."""
    edges = np.empty(len(depth) + 1, dtype=float)
    edges[1:-1] = (depth[:-1] + depth[1:]) / 2.0
    edges[0] = depth[0] - (depth[1] - depth[0]) / 2.0
    edges[-1] = depth[-1] + (depth[-1] - depth[-2]) / 2.0
    return np.abs(np.diff(edges))


def depth_agreement(aligned: dict, boundary_tolerance_m: float | None = None) -> dict:
    """Fraction of the profile where interpretation and reference agree."""
    depth = np.asarray(aligned["depth"], dtype=float)
    gt = np.asarray(aligned["ground_truth_ids"])
    interp = np.asarray(aligned["interpreted_ids"])
    weight = _sample_thickness(depth)
    total = float(weight.sum())

    match = interp == gt
    result = {
        "cpt_id": aligned["cpt_id"],
        "profile_length_m": total,
        "agreement": float(weight[match].sum() / total),
        "disagreement_m": float(weight[~match].sum()),
        "unweighted_point_agreement": float(match.mean()),
    }

    per_class = {}
    for soil_id, name in ID_TO_LABEL.items():
        sel = gt == soil_id
        if sel.any():
            per_class[name] = float(weight[sel & match].sum() / weight[sel].sum())
    result["per_class_recall"] = per_class

    if boundary_tolerance_m is not None:
        z = np.abs(depth)
        bounds = z[1:][gt[1:] != gt[:-1]]
        if len(bounds):
            near = np.min(np.abs(z[:, None] - bounds[None, :]), axis=1) <= boundary_tolerance_m
        else:
            near = np.zeros(len(z), dtype=bool)
        excused = ~match & near
        result["boundary_tolerance_m"] = boundary_tolerance_m
        result["agreement_boundary_tolerant"] = float(
            weight[match | excused].sum() / total
        )
        result["disagreement_at_boundary_m"] = float(weight[excused].sum())
        result["disagreement_interior_m"] = float(weight[~match & ~near].sum())

    return result


def coverage_report(cpt_id: int, segments: list[pd.DataFrame]) -> dict:
    """How well each segment is resolved by the coarse grid."""
    rows = []
    for i, seg in enumerate(segments, start=1):
        z0 = abs(float(seg["depth"].iloc[0]))
        z1 = abs(float(seg["depth"].iloc[-1]))
        thickness = abs(z1 - z0)
        rows.append(
            {
                "segment": i,
                "thickness_m": thickness,
                "cells_spanned": thickness * CELLS_PER_M,
                "weakly_resolved": thickness * CELLS_PER_M < 2.0,
            }
        )
    return {
        "cpt_id": cpt_id,
        "cell_size_m": 1.0 / CELLS_PER_M,
        "n_segments": len(rows),
        "n_weakly_resolved": sum(r["weakly_resolved"] for r in rows),
        "segments": rows,
    }


if __name__ == "__main__":
    from cpt_segmentation import load_cpt, segment_ic

    grid = load_grid()
    print(f"Grid: {grid.shape} = {grid.shape[0] / CELLS_PER_M:.0f} m deep "
          f"x {grid.shape[1]} m wide, {CELLS_PER_M} cells/m vertically")
    print(f"Classes: {ID_TO_LABEL}")

    cpt_id = 1
    df = load_cpt()
    segments = segment_ic(df, pen=5.0)
    labels = segment_labels(cpt_id, segments)
    cov = coverage_report(cpt_id, segments)

    print(f"\nCPT{cpt_id} -> grid column {column_for(cpt_id)}, "
          f"{len(segments)} segments")
    for row, label in zip(cov["segments"], labels):
        flag = "  (weak: <2 cells)" if row["weakly_resolved"] else ""
        print(f"  [{row['segment']:02d}] {row['thickness_m']:6.3f} m "
              f"({row['cells_spanned']:4.1f} cells)  {label}{flag}")
    print(f"\n{cov['n_weakly_resolved']}/{cov['n_segments']} segments are weakly resolved "
          f"by the {cov['cell_size_m']:.3f} m grid.")
