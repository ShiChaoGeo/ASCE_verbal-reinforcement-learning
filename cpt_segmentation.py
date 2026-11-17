"""Load CPTu data from cpt_x.xlsx and segment Ic with PELT change-point detection."""

from pathlib import Path

import pandas as pd
import ruptures as rpt


DATA_DIR = Path(__file__).parent

CPT_FILE = DATA_DIR / "cpt_1.xlsx"


def load_cpt(path: Path = CPT_FILE) -> pd.DataFrame:
    """Read Sheet1 and return the CPTu records (12 columns)."""
    return pd.read_excel(path, sheet_name="Sheet1")


def data_gaps(path: Path = CPT_FILE,
              min_thickness_m: float = 0.10) -> list[tuple[float, float]]:
    """Depth intervals, positive down, where the record is missing."""
    depth = pd.read_excel(path, sheet_name="Sheet1")["depth"].to_numpy(float)
    missing = pd.isna(depth)
    if not missing.any():
        return []
    down = -depth
    deepest = float(pd.Series(down).max())
    out, start = [], None
    for i, gone in enumerate(missing):
        if gone and start is None:
            start = i
        elif not gone and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(missing) - 1))

    spans = []
    for a, b in out:
        top = float(down[a - 1]) if a > 0 else 0.0
        base = float(down[b + 1]) if b + 1 < len(down) else deepest
        if base - top >= min_thickness_m:
            spans.append((top, base))
    return spans


def detect_ic_breakpoints(df: pd.DataFrame, pen: float = 5.0) -> list[int]:
    """Run PELT (L2) change-point detection on the Ic column."""
    ic = df["Ic"].dropna().to_numpy()
    algo = rpt.Pelt(model="l2").fit(ic)
    return algo.predict(pen=pen)


def _segment_thickness_m(seg: pd.DataFrame) -> float:
    """Positive-down thickness (m) from workbook depths (negative downward)."""
    z0 = float(-seg["depth"].iloc[0])
    z1 = float(-seg["depth"].iloc[-1])
    return abs(z1 - z0)


def merge_thin_segments(
    segments: list[pd.DataFrame],
    min_thickness_m: float = 1 / 6,
) -> list[pd.DataFrame]:
    """Merge any segment thinner than ``min_thickness_m`` into the previous one."""
    if not segments:
        return []

    merged: list[pd.DataFrame] = [segments[0].copy()]
    for seg in segments[1:]:
        if _segment_thickness_m(seg) < min_thickness_m:
            merged[-1] = pd.concat([merged[-1], seg], ignore_index=True)
        else:
            merged.append(seg.copy())
    return merged


def segment_ic(
    df: pd.DataFrame,
    pen: float = 5.0,
    min_thickness_m: float = 1 / 6,
) -> list[pd.DataFrame]:
    """Split CPT data into sequential unlabeled segments using Ic change points."""
    valid = df.loc[df["Ic"].notna()].reset_index(drop=True)
    breakpoints = detect_ic_breakpoints(df, pen=pen)

    starts = [0] + breakpoints[:-1]
    raw = [valid.iloc[start:end].copy() for start, end in zip(starts, breakpoints)]
    return merge_thin_segments(raw, min_thickness_m=min_thickness_m)


if __name__ == "__main__":
    df = load_cpt()

    print(f"File: {CPT_FILE.name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Depth range: {df['depth'].max():.3f} to {df['depth'].min():.3f} m")

    valid = df.loc[df["Ic"].notna()].reset_index(drop=True)
    cpt_data = valid["Ic"].to_numpy()
    cpt_depth = valid["depth"].to_numpy()

    breakpoints = detect_ic_breakpoints(df, pen=5)
    segments = segment_ic(df, pen=5)

    print(f"\nIc change points (PELT, pen=5): {len(breakpoints) - 1} interior + end")
    print(f"Breakpoints (indices into Ic-valid rows): {breakpoints}")
    print(f"Sequential unlabeled segments (after <0.18 m merge): {len(segments)}")

    print("\nSegments:")
    for i, seg in enumerate(segments):
        print(
            f"  [{i:02d}] depth {seg['depth'].iloc[0]:.3f} \u2192 {seg['depth'].iloc[-1]:.3f} m "
            f"(n={len(seg)}, cols={list(seg.columns)})"
        )
