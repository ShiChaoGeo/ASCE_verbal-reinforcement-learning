"""Environment: one CPTu sounding, observed as a set of unlabeled segments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

from cpt_segmentation import load_cpt, segment_ic
from segment_features import segment_feature_row

DATA_DIR = Path(__file__).resolve().parent.parent

ACTION_SPACE = (
    "fill",
    "disturbed_marine_deposit",
    "marine_deposit",
    "alluvial_sand",
    "alluvial_clay",
)

OBSERVED_FEATURES = (
    "Ic_median",
    "Ic_roughness",
    "Ic_robust_COV",
    "Bq_median",
    "Bq_robust_COV",
    "Bq_roughness",
    "qc_roughness",
    "fs_roughness",
)


_DEPTH_RE = re.compile(r"([-\d.]+)\s*-\s*([-\d.]+)")


def parse_depth_label(label: str) -> tuple[float | None, float | None]:
    """Pull (top, base) in metres out of a card's ``depth_label``."""
    match = _DEPTH_RE.search(label or "")
    if not match:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def cpt_path(cpt_id: int) -> Path:
    """Training sounding: ``cpt_{id}.xlsx``."""
    return DATA_DIR / f"cpt_{cpt_id}.xlsx"


def _valid_index(suffix: str) -> bool:
    """``7`` and ``v7`` are indices; anything else is not."""
    m = re.fullmatch(r"([a-z]?)([0-9]+)", suffix)
    return bool(m) and int(m.group(2)) > 0


def normalize_application_key(arg: str | int) -> str:
    """Return ``unknown_i`` for application soundings named ``cpt_unknown_i.xlsx``."""
    s = str(arg).strip().lower()
    if s.startswith("unknown_"):
        suffix = s[len("unknown_"):]
        if _valid_index(suffix):
            return f"unknown_{suffix}"
        raise ValueError(f"Application CPT key must look like unknown_1, got {arg!r}")
    if _valid_index(s):
        return f"unknown_{s}"
    raise ValueError(
        f"Application CPT key must be an index (e.g. 1) or unknown_1, got {arg!r}"
    )


MIN_THICKNESS_M = 1 / 6


def _application_dirs() -> list[Path]:
    return [DATA_DIR] + sorted(p for p in DATA_DIR.glob("case_study_*") if p.is_dir())


def application_cpt_path(key: str) -> Path:
    """Application sounding ``cpt_unknown_i.xlsx``, wherever it is filed."""
    key = normalize_application_key(key)
    name = f"cpt_{key}.xlsx"
    for folder in _application_dirs():
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return DATA_DIR / name


def available_cpt_ids() -> list[int]:
    """Sorted training CPT ids for which ``cpt_{id}.xlsx`` is present."""
    ids: list[int] = []
    for path in DATA_DIR.glob("cpt_*.xlsx"):
        stem = path.stem
        if not stem.startswith("cpt_"):
            continue
        suffix = stem[4:]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def available_application_keys() -> list[str]:
    """Sorted application keys ``unknown_i`` for ``cpt_unknown_i.xlsx`` files."""
    keys: set[str] = set()
    for folder in _application_dirs():
        for path in folder.glob("cpt_unknown_*.xlsx"):
            keys.add(path.stem.removeprefix("cpt_"))
    return sorted(keys, key=lambda k: [int(t) if t.isdigit() else t
                                       for t in re.split(r"(\d+)", k)])


@dataclass
class State:
    """What the policy conditions on for one episode."""

    cpt_id: int | str
    cards: list[dict]
    segments: list[pd.DataFrame]
    frame: pd.DataFrame

    @property
    def n_segments(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"State(cpt={self.cpt_id}, {self.n_segments} segments)"


def _to_card(row: pd.Series, cpt_id: int | str) -> dict:
    """One segment as the policy sees it: features, no label."""
    return {
        "cpt_id": cpt_id,
        "segment": int(row["segment"]),
        "depth_label": row["depth_label"],
        "n_points": int(row["n_points"]),
        "features": {
            k: (None if pd.isna(row[k]) else float(row[k])) for k in OBSERVED_FEATURES
        },
        "robertson_behavior": row["robertson_behavior"],
    }


def observe(cpt_id: int, pen: float = 5.0) -> State:
    """Load a training sounding and reduce it to unlabeled segment feature cards."""
    path = cpt_path(cpt_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Place cpt_{cpt_id}.xlsx in the project root first."
        )
    return _observe_file(path, cpt_id, pen=pen)


def observe_application(key: str | int, pen: float = 5.0,
                        min_thickness_m: float = MIN_THICKNESS_M) -> State:
    """Load an application sounding ``cpt_unknown_i.xlsx``."""
    app_key = normalize_application_key(key)
    path = application_cpt_path(app_key)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Place cpt_{app_key}.xlsx in a case_study_* folder."
        )
    return _observe_file(path, app_key, pen=pen,
                         min_thickness_m=min_thickness_m)


def _observe_file(path: Path, cpt_id: int | str, pen: float,
                  min_thickness_m: float = MIN_THICKNESS_M) -> State:
    """Shared loader for training and application soundings."""
    frame = load_cpt(path)
    segments = segment_ic(frame, pen=pen, min_thickness_m=min_thickness_m)
    rows = [segment_feature_row(seg, i) for i, seg in enumerate(segments, start=1)]
    cards = [_to_card(pd.Series(r), cpt_id) for r in rows]
    return State(cpt_id=cpt_id, cards=cards, segments=segments, frame=frame)
