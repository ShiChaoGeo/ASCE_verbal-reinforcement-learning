"""Apply a finished rulebook to a folder of unlabelled soundings."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vrl.environment import (
    ACTION_SPACE,
    MIN_THICKNESS_M,
    application_cpt_path,
    observe_application,
    parse_depth_label,
)
from vrl.policy import ADMISSIBLE_BANDS, PHASE_OF, HardConstraintUnmet, Policy, act
from vrl.providers import DEFAULT_PROVIDER
from vrl.trajectory import RUNS_DIR

ROOT = Path(__file__).resolve().parents[1]
APPLY_DIR = RUNS_DIR / "apply"
STATE_FILE = APPLY_DIR / "state.json"

DRAWS = 3
PHASE_LABELS = {1: [l for l in ACTION_SPACE if PHASE_OF[l] == 1],
                2: [l for l in ACTION_SPACE if PHASE_OF[l] == 2],
                3: [l for l in ACTION_SPACE if PHASE_OF[l] == 3]}


def case_folders() -> list[str]:
    """Directories holding application soundings, nearest first."""
    found: list[str] = []
    if list(ROOT.glob("cpt_unknown_*.xlsx")):
        found.append(".")
    for child in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if child.name.startswith((".", "_")) or child.name in {"vrl", "vrl_runs"}:
            continue
        if list(child.glob("cpt_unknown_*.xlsx")):
            found.append(child.name)
    return found


def soundings_in(folder: str) -> list[str]:
    """Application keys held by one folder, in numeric order."""
    base = ROOT if folder in (".", "") else ROOT / folder
    numbered = []
    for path in base.glob("cpt_unknown_*.xlsx"):
        key = path.stem.removeprefix("cpt_")
        m = re.fullmatch(r"unknown_([a-z]?)([0-9]+)", key.lower())
        if m:
            numbered.append(((m.group(1), int(m.group(2))), key.lower()))
    return [k for _, k in sorted(numbered)]


def available_rulebooks() -> list[str]:
    """Rulebook versions on disk, the search versions in numeric order."""
    versions = []
    for path in Policy.path_for("v0").parent.glob("cpt_lithology_rulebook_*.md"):
        versions.append(path.stem.removeprefix("cpt_lithology_rulebook_"))

    def key(v: str) -> tuple[int, int]:
        m = re.fullmatch(r"([a-z]+)(\d+)", v)
        return (0 if v == "v0" else 1, int(m.group(2)) if m else 0)

    return sorted(versions, key=key)


def _votes(draws: list[list[str]], n: int) -> list[dict[str, int]]:
    counts = [{label: 0 for label in ACTION_SPACE} for _ in range(n)]
    for labels in draws:
        for i, label in enumerate(labels):
            counts[i][label] += 1
    return counts


def semantic_entropy(votes: dict[str, int]) -> float:
    """Shannon entropy of one unit's label distribution, in nats."""
    n = sum(votes.values())
    if n <= 0:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in votes.values() if c)
    return abs(h)


def _admissible(phase: int, card: dict | None) -> tuple[str, ...]:
    """The labels a unit may carry in a phase, given how it behaves."""
    labels = PHASE_LABELS[phase]
    band = (card or {}).get("robertson_behavior", "")
    if not band or band == "unknown":
        return labels
    kept = tuple(l for l in labels if band in ADMISSIBLE_BANDS.get(l, {band}))
    return kept or labels


def consensus(draws: list[list[str]], cards: list[dict] | None = None) -> list[dict]:
    """The admissible profile carrying the most votes, unit by unit."""
    if not draws:
        return []
    n = len(draws[0])
    counts = _votes(draws, n)
    total = len(draws)
    if cards is not None and len(cards) != n:
        cards = None
    allowed = [{p: _admissible(p, cards[i] if cards else None) for p in (1, 2, 3)}
               for i in range(n)]

    best_in = [{p: max(counts[i][l] for l in allowed[i][p]) for p in (1, 2, 3)}
               for i in range(n)]
    pre = {p: [0] * (n + 1) for p in (1, 2, 3)}
    for p in (1, 2, 3):
        for i in range(n):
            pre[p][i + 1] = pre[p][i] + best_in[i][p]

    def labels_for(a: int, b: int) -> list[str]:
        out = []
        for i in range(n):
            phase = 1 if i < a else (2 if i < b else 3)
            out.append(max(allowed[i][phase], key=lambda l: counts[i][l]))
        return out

    def pair_held(labels: list[str]) -> bool:
        """The alluvial interval exists and carries both of its soils."""
        alluvial = {l for l in labels if PHASE_OF[l] == 3}
        return {"alluvial_sand", "alluvial_clay"} <= alluvial

    ranked = sorted(
        ((pre[1][a] + (pre[2][b] - pre[2][a]) + (pre[3][n] - pre[3][b]), a, b)
         for a in range(n + 1) for b in range(a, n + 1)),
        key=lambda s: -s[0])
    cuts = (ranked[0][1], ranked[0][2])
    for _, a, b in ranked:
        if pair_held(labels_for(a, b)):
            cuts = (a, b)
            break
    a, b = cuts
    chosen = labels_for(a, b)

    out = []
    for i in range(n):
        label = chosen[i]
        modal = max(ACTION_SPACE, key=lambda l: counts[i][l])
        votes = {l: c for l, c in counts[i].items() if c}
        out.append({
            "segment": i + 1,
            "label": label,
            "probability": counts[i][label] / total,
            "votes": votes,
            "entropy": semantic_entropy(votes),
            "n_draws": total,
            "unanimous": counts[i][label] == total,
            "constrained": modal != label,
        })
    return out


@dataclass
class InterpretState:
    """What to interpret, and what has been interpreted already."""

    folder: str
    policy_version: str
    draws: int = DRAWS
    append: bool = False
    keys: list[str] = field(default_factory=list)
    queue: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    error: str = ""

    @classmethod
    def load(cls) -> "InterpretState | None":
        if not STATE_FILE.is_file():
            return None
        try:
            return cls(**json.loads(STATE_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(STATE_FILE)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["n_done"] = len([k for k in self.done if k in self.queue])
        d["n_total"] = len(self.queue)
        return d


def output_dir(folder: str, policy_version: str) -> Path:
    return APPLY_DIR / folder.replace(".", "root") / policy_version


def _earlier_results(folder: str, policy_version: str, skip: list[str]) -> dict:
    """Results already on disk for this folder and rulebook, minus ``skip``."""
    out: dict = {}
    base = output_dir(folder, policy_version)
    if not base.is_dir():
        return out
    for path in base.glob("*.json"):
        key = path.stem
        if key in skip:
            continue
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[key] = {k: r[k] for k in ("n_segments", "draws_used", "agreement",
                                      "unanimous", "constrained", "profile")
                    if k in r}
    return out


def start(folder: str, policy_version: str, draws: int = DRAWS,
          only: list[str] | None = None, append: bool = False) -> InterpretState:
    """Begin an interpretation of a folder, or of named soundings within it."""
    keys = soundings_in(folder)
    if not keys:
        raise RuntimeError(f"no cpt_unknown_*.xlsx in {folder!r}")
    queue = list(keys)
    if only:
        unknown = [k for k in only if k not in keys]
        if unknown:
            raise RuntimeError(f"{', '.join(unknown)} not in {folder!r}")
        queue = [k for k in keys if k in only]
    Policy.load(policy_version)

    carried = _earlier_results(folder, policy_version, skip=queue)
    state = InterpretState(
        folder=folder, policy_version=policy_version, draws=int(draws),
        append=bool(append),
        keys=keys, queue=queue, done=sorted(carried, key=keys.index),
        results=carried,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    state.save()
    return state


def interpret_one(key: str, policy: Policy, draws: int, *, provider: str,
                  model: str | None, pen: float,
                  prior: dict | None = None,
                  min_thickness: float = MIN_THICKNESS_M) -> dict:
    """Segment one sounding, interpret it `draws` times, and reduce the draws."""
    state = observe_application(key, pen=pen, min_thickness_m=min_thickness)
    cards = state.cards

    labels_per_draw: list[list[str]] = []
    actions: list[list[dict]] = []
    failures: list[str] = []
    done_before = 0
    if prior and prior.get("n_segments") == len(cards):
        labels_per_draw = [list(x) for x in prior.get("draw_labels", [])]
        actions = list(prior.get("actions", []))
        failures = list(prior.get("failures", []))
        done_before = len(labels_per_draw)

    for d in range(done_before + 1, done_before + draws + 1):
        try:
            action = act(policy, cards, provider=provider, model=model)
        except HardConstraintUnmet as exc:
            failures.append(f"draw {d}: {exc}")
            continue
        by_segment = {int(a["segment"]): a for a in action}
        labels = [by_segment[i + 1]["predicted_label"] for i in range(len(cards))]
        labels_per_draw.append(labels)
        actions.append(action)

    profile = consensus(labels_per_draw, cards)
    for unit, card in zip(profile, cards):
        top, base = parse_depth_label(card.get("depth_label", ""))
        unit["top"], unit["base"] = top, base
        unit["depth_label"] = card.get("depth_label", "")

    result = {
        "cpt": key,
        "file": str(application_cpt_path(key).relative_to(ROOT)),
        "n_segments": len(cards),
        "draws_requested": done_before + draws,
        "draws_used": len(labels_per_draw),
        "failures": failures,
        "cards": cards,
        "draw_labels": labels_per_draw,
        "profile": profile,
        "agreement": (sum(u["probability"] for u in profile) / len(profile)
                      if profile else 0.0),
        "unanimous": sum(1 for u in profile if u["unanimous"]),
        "constrained": sum(1 for u in profile if u["constrained"]),
        "actions": actions,
    }
    return result


def run(*, provider: str = DEFAULT_PROVIDER, model: str | None = None,
        pen: float = 5.0, min_thickness: float = MIN_THICKNESS_M) -> dict:
    """Interpret every sounding of the folder the state names."""
    state = InterpretState.load()
    if state is None:
        raise RuntimeError("No interpretation in progress. Start one first.")

    policy = Policy.load(state.policy_version)
    out_dir = output_dir(state.folder, state.policy_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = state.queue or state.keys

    for key in queue:
        if key in state.done:
            continue
        prior = None
        if state.append:
            path = out_dir / f"{key}.json"
            if path.is_file():
                prior = json.loads(path.read_text(encoding="utf-8"))
        result = interpret_one(key, policy, state.draws, provider=provider,
                               model=model, pen=pen, prior=prior,
                               min_thickness=min_thickness)
        (out_dir / f"{key}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if key not in state.done:
            state.done.append(key)
        state.results[key] = {k: result[k] for k in
                              ("n_segments", "draws_used", "agreement",
                               "unanimous", "constrained", "profile")}
        state.save()

    state.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.save()
    return state.as_dict()
