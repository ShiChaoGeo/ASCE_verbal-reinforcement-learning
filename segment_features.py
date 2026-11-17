"""Segment-level features of a CPTu sounding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpt_segmentation import CPT_FILE, load_cpt, segment_ic

DATA_DIR = Path(__file__).parent
OUT_CSV = DATA_DIR / "cpt1_segment_features.csv"


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def iqr(x: np.ndarray) -> float:
    """Interquartile range: Q75 - Q25."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def robust_cov(x: np.ndarray, scale_constant: float = 1.4826) -> float:
    """Robust coefficient of variation: scale_constant * MAD / |median|."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    if abs(med) < 1e-12:
        return float("nan")
    return float(scale_constant * mad(x) / abs(med))


def roughness(x: np.ndarray) -> float:
    """How far a reading sits from its neighbours, relative to the level."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    med = float(np.median(x))
    if abs(med) < 1e-12:
        return float("nan")
    return float(np.median(np.abs(np.diff(x))) / abs(med))


def robertson_behavior(ic_median: float) -> str:
    """Robertson behaviour band from the Ic median."""
    if not np.isfinite(ic_median):
        return "unknown"
    if ic_median < 2.05:
        return "sand-like"
    if ic_median < 2.60:
        return "between sand-like and clay-like, closer to sandy / sandy-silty"
    if ic_median < 2.95:
        return "between sand-like and clay-like / silty"
    return "clay-like"


def segment_feature_row(seg: pd.DataFrame, segment_id: int) -> dict:
    depth = seg["depth"].to_numpy(dtype=float)
    ic = seg["Ic"].to_numpy(dtype=float)
    bq = seg["Bq"].to_numpy(dtype=float)
    qc = seg["qc"].to_numpy(dtype=float)
    fs = seg["fs"].to_numpy(dtype=float)

    z0 = float(-depth[0])
    z1 = float(-depth[-1])
    if z1 < z0:
        z0, z1 = z1, z0

    ic_median = float(np.nanmedian(ic))
    bq_median = float(np.nanmedian(bq))

    return {
        "segment": segment_id,
        "n_points": int(len(seg)),
        "depth_top_m": z0,
        "depth_base_m": z1,
        "thickness_m": z1 - z0,
        "Ic_median": ic_median,
        "Ic_robust_COV": robust_cov(ic),
        "Bq_median": bq_median,
        "Bq_IQR": iqr(bq),
        "Bq_robust_COV": robust_cov(bq),
        "qc_median": float(np.nanmedian(qc)),
        "qc_robust_COV": robust_cov(qc),
        "fs_median": float(np.nanmedian(fs)),
        "fs_robust_COV": robust_cov(fs),
        "Ic_roughness": roughness(ic),
        "Bq_roughness": roughness(bq),
        "qc_roughness": roughness(qc),
        "fs_roughness": roughness(fs),
        "robertson_behavior": robertson_behavior(ic_median),
        "depth_label": f"{z0:.3f}-{z1:.3f} m",
    }


def compute_segment_features(
    df: pd.DataFrame | None = None,
    pen: float = 5.0,
) -> pd.DataFrame:
    """Return one feature row per PELT Ic segment (shallow → deep)."""
    if df is None:
        df = load_cpt()
    segments = segment_ic(df, pen=pen)
    rows = [segment_feature_row(seg, i) for i, seg in enumerate(segments, start=1)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    feats = compute_segment_features(pen=5.0)
    feats.to_csv(OUT_CSV, index=False, float_format="%.6f")

    print(f"Source: {CPT_FILE.name}")
    print(f"Segments: {len(feats)}")
    print(f"Wrote: {OUT_CSV.name}\n")
    cols = [
        "segment",
        "depth_label",
        "Ic_median",
        "Ic_robust_COV",
        "Bq_median",
        "Bq_IQR",
        "Bq_robust_COV",
        "qc_robust_COV",
        "fs_robust_COV",
    ]
    print(feats[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
