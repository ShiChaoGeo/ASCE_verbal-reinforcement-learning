"""Rulebook search: v0 plus at most three learned practices, revised each round."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vrl.batch import interpret_batch, reflect_batch, score_batch
from vrl.policy import Policy, hard_constraint_violations
from vrl.providers import DEFAULT_PROVIDER, call_provider
from vrl.trajectory import RUNS_DIR

SEARCH_DIR = RUNS_DIR / "search"
STATE_FILE = SEARCH_DIR / "state.json"

SUBSECTIONS = {
    "geological": "Geological knowledge",
    "geotechnical": "Geotechnical knowledge",
}


def compose(base_version: str, rules: list[dict], new_version: str) -> Policy:
    """Write ``base + the active rules`` as a new rulebook version."""
    text = Policy.load(base_version).text

    for r in rules:
        if not r.get("id"):
            raise ValueError("every rule needs an id so its citations can be counted")

    for subsection in ("geological", "geotechnical"):
        bullets = [r for r in rules if r["subsection"] == subsection]
        if not bullets:
            continue
        title = SUBSECTIONS[subsection]
        found = re.search(rf"(?m)^(#{{2,3}}) {re.escape(title)}\s*$", text)
        if not found:
            raise ValueError(f"{base_version} has no '{title}' section to add to.")

        start = found.end()
        nxt = re.search(r"(?m)^#{1,6} ", text[start:])
        end = start + nxt.start() if nxt else len(text)

        body = text[start:end].strip("\n").rstrip()
        parts = [text[:start].rstrip("\n"), "\n\n"]
        if body:
            parts += [body, "\n"]
        parts += ["\n".join(f"- [{r['id']}] " + " ".join(r["text"].split())
                             for r in bullets), "\n\n"]
        text = "".join(parts) + text[end:]

    text = text.replace(
        f"# CPT Lithology Interpretation Rulebook {base_version}",
        f"# CPT Lithology Interpretation Rulebook {new_version}", 1,
    )

    path = Policy.path_for(new_version)
    if path.is_file():
        raise FileExistsError(f"{path} already exists; policy versions are append-only.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return Policy(version=new_version, text=text)


MAX_RULES = 3

HARM_WEIGHT = 2


def constraint_violations(trajectories) -> dict[int, list[dict]]:
    """Breaches that survived the interpreter's own retries, by sounding."""
    found: dict[int, list[dict]] = {}
    for trajectory in trajectories:
        hits = hard_constraint_violations(trajectory.load_action(), trajectory.load_state())
        if hits:
            found[trajectory.cpt_id] = hits
    return found


def citation_stats(trajectories, per_segment: dict, active: list[dict],
                   violations: dict | None = None) -> dict[str, dict]:
    """How each practice fared where it was actually invoked."""
    correct = {(c, int(g["segment"])): g["correct"]
               for c, segs in per_segment.items() for g in segs}
    stats = {r["id"]: {"cited": 0, "right": 0, "wrong": 0, "wrong_units": [],
                       "breaches": 0, "breach_units": []}
             for r in active}
    broken = {(cpt, h["unit"]): h for cpt, hits in (violations or {}).items() for h in hits}

    for trajectory in trajectories:
        for item in trajectory.load_action():
            unit = int(item["segment"])
            applied = " ".join(item.get("rules_applied") or [])
            for rule_id in stats:
                if f"[{rule_id}]" not in applied:
                    continue
                s = stats[rule_id]
                s["cited"] += 1
                if correct.get((trajectory.cpt_id, unit), True):
                    s["right"] += 1
                else:
                    s["wrong"] += 1
                    s["wrong_units"].append(f"CPT{trajectory.cpt_id} unit {unit}")
                breach = broken.get((trajectory.cpt_id, unit))
                if breach:
                    s["breaches"] += 1
                    s["breach_units"].append(
                        f"CPT{trajectory.cpt_id} unit {unit} ({breach['constraint']})")

    for s in stats.values():
        s["precision"] = s["right"] / s["cited"] if s["cited"] else None
        s["reward"] = s["right"] - HARM_WEIGHT * s["wrong"] - HARM_WEIGHT * s["breaches"]
    return stats


def advice(stats: dict[str, dict], active: list[dict]) -> str:
    """What the citations point at, for the writer to weigh."""
    if not active:
        return ""
    breaching = [r["id"] for r in active if stats.get(r["id"], {}).get("breaches")]
    harmful = [r["id"] for r in active if stats.get(r["id"], {}).get("wrong")
               and r["id"] not in breaching]
    inert = [r["id"] for r in active if not stats.get(r["id"], {}).get("cited")]
    earning = [r["id"] for r in active
               if stats.get(r["id"], {}).get("cited") and not stats[r["id"]]["wrong"]]

    lines = []
    if breaching:
        lines.append(f"- {', '.join(breaching)} was cited where a [Hard constraint] "
                     "broke. That is the failure to address first.")
    if harmful:
        lines.append(f"- {', '.join(harmful)} was cited on wrong labels.")
    if inert:
        lines.append(f"- {', '.join(inert)} was never cited: it is doing nothing.")
    if earning:
        lines.append(f"- {', '.join(earning)} was cited and right throughout. "
                     "Returning it verbatim is probably the right answer for it.")
    return "\n".join(lines) + "\n" if lines else ""


RULE_SYSTEM = """\
You are maintaining the practices that an engineer's CPTu lithology rulebook is
missing.

The rulebook is the engineer's own and does not change. Alongside it stand at
most three practices, and this round you return the set as it should now be:
keep a practice unchanged by returning it verbatim under its own id, rewrite one
by returning new wording under its id, drop one by leaving it out, and introduce
one with the id "new". Change as much or as little as the evidence warrants —
one word or all of them.

You are told, for each standing practice, the units where the interpreter cited
it and whether the label was right, wrong, or broke a [Hard constraint]. That is
the evidence:

- Cited and wrong is a practice at fault. Rewriting it usually means saying more
  precisely WHEN it applies, not widening it to cover more.
- Cited on a unit where a constraint broke is the worst outcome available: a
  practice may refine a constraint, never override it.
- Never cited is a practice doing nothing. Drop it or replace it; a slot held by
  a dead practice is a slot the errors cannot reach.
- Cited and right is a practice earning its place. Returning it verbatim is the
  right answer for it, and rewriting a working practice to chase an unrelated
  error is how a set degrades.
- A change that repairs one sounding while breaking another is worse than none.

Write each practice as a statement about evidence, applicable to profiles you
have never seen: name no sounding, no unit, no depth of any profile. Thresholds
on features are welcome — which feature, which direction, what magnitude
relative to what.

Geological knowledge concerns depositional origin and stratigraphic history.
Geotechnical knowledge concerns CPTu response and its interpretation.
"""

RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "practices": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_RULES,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": (
                            "The id of the standing practice this replaces, or "
                            "the word new for one being introduced."
                        ),
                    },
                    "subsection": {"type": "string",
                                   "enum": ["geological", "geotechnical"]},
                    "rule": {"type": "string",
                             "description": ("The practice, as ONE sentence. No "
                                             "sounding, unit or depth of this dataset.")},
                    "rationale": {"type": "string",
                                  "description": "Which errors it accounts for."},
                },
                "required": ["id", "subsection", "rule", "rationale"],
                "additionalProperties": False,
            },
        },
        "changes": {
            "type": "string",
            "description": (
                "What changed in this set and why, in a sentence or two: which "
                "practices were kept, rewritten, dropped or added."
            ),
        },
    },
    "required": ["practices", "changes"],
    "additionalProperties": False,
}


SEED_SYSTEM = """\
You are writing the practices that an engineer's CPTu lithology rulebook is
missing.

The rulebook you are given is the engineer's own. It is not wrong; it is
underdetermined at a few steps, so the same data can be read more than one way.

Write the practices it is missing, as many as the errors below actually call for
and AT MOST three. Let the errors decide the number: if they all come of one
thing, write one. Padding to three buys nothing and costs the room a later round
will want.

- Each must cover a different way the interpretation went wrong. Two wordings of
  one idea are one practice, not two.
- None may restate or contradict another, and none may restate the rulebook.

A practice may refine a [Hard constraint] — say when it bites, how to recognise
the case — but it may never override one. If applying a practice would break a
constraint, the practice does not apply there, and its wording must say so.

Write each as a statement about evidence, applicable to profiles you have never
seen: name no sounding, no unit, no depth of any profile. Thresholds on features
are welcome — which feature, which direction, what magnitude relative to what.

Geological knowledge concerns depositional origin and stratigraphic history.
Geotechnical knowledge concerns CPTu response and its interpretation.
"""

SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "practices": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_RULES,
            "items": {
                "type": "object",
                "properties": {
                    "subsection": {"type": "string",
                                   "enum": ["geological", "geotechnical"]},
                    "rule": {"type": "string",
                             "description": ("The practice, as ONE sentence. No "
                                             "sounding, unit or depth of this dataset.")},
                    "rationale": {"type": "string",
                                  "description": "Which of the errors it accounts for."},
                },
                "required": ["subsection", "rule", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["practices"],
    "additionalProperties": False,
}


def build_seed_prompt(base_version: str, errors: list[dict]) -> str:
    return "".join([
        f"# The engineer's rulebook ({base_version})\n",
        Policy.load(base_version).text,
        _error_block(errors, "Errors this rulebook made"),
        f"\nWrite the practices — at most {MAX_RULES}.\n",
    ])


def _error_block(errors: list[dict], title: str) -> str:
    if not errors:
        return ""
    lines = [f"\n# {title}\n"]
    for e in errors:
        lines.append(
            f"- round {e['round']} · rule in force: {e['rule_id'] or 'none (base rulebook)'}"
            f"\n  CPT{e['cpt_id']} unit {e['segment']} ({e['depth_label']})"
            f"\n  interpreted as {e['predicted_label']}, engineer says {e['engineer_label']}"
            f"\n  features: {json.dumps(e.get('features', {}), ensure_ascii=False)}"
        )
    return "\n".join(lines) + "\n"


def _rules_block(title: str, rules: list[dict], stats: dict | None = None) -> str:
    if not rules:
        return ""
    lines = [f"\n# {title}\n"]
    for r in rules:
        head = (f"- {r['id']} ({r['subsection']}, added in round {r['round']}"
                + (f", revised {r['revisions']}x" if r.get("revisions") else "") + ")")
        entry = f"{head}: {r['text']}\n  reason given: {r['rationale']}"
        s = (stats or {}).get(r["id"])
        if s:
            if s["cited"]:
                entry += (f"\n  this round it was cited on {s['cited']} unit(s): "
                          f"{s['right']} right, {s['wrong']} wrong")
                if s["wrong_units"]:
                    entry += " — wrong at " + ", ".join(s["wrong_units"])
                if s.get("breaches"):
                    entry += ("\n  it was cited on " + str(s["breaches"])
                              + " unit(s) where a [Hard constraint] was broken: "
                              + ", ".join(s["breach_units"]))
            else:
                entry += "\n  this round it was never cited"
            if r.get("cumulative_reward") is not None:
                entry += f"\n  cumulative reward: {r['cumulative_reward']:+d}"
        lines.append(entry)
    return "\n".join(lines) + "\n"


def build_rule_prompt(base_version: str, active: list[dict], this_round: list[dict],
                      memory: list[dict], retired: list[dict], stats: dict) -> str:
    parts = [
        f"# The engineer's rulebook ({base_version})\n",
        Policy.load(base_version).text,
        _rules_block("Practices standing alongside it", active, stats),
    ]
    earlier = [e for e in memory if e not in this_round]
    parts += [
        _error_block(this_round, "Errors in this round"),
        _error_block(earlier, "Errors remembered from earlier rounds"),
        _rules_block("Wordings already tried and dropped", retired),
        "\n# What the citations point at\n\n" + (advice(stats, active) or
        "- Nothing stands out; read the errors.\n"),
        f"\nReturn the set as it should now be, at most {MAX_RULES} practices. "
        "Keep a practice by returning it verbatim under its id, rewrite it under "
        "its id, drop it by leaving it out, or introduce one with the id \"new\".\n",
    ]
    return "".join(parts)


@dataclass
class SearchState:
    cpt_ids: list[int]
    base_version: str = "v0"
    max_rounds: int = 15
    current_version: str = "v0"
    active: list[dict] = field(default_factory=list)
    retired: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    memory: list[dict] = field(default_factory=list)
    started_at: str = ""

    @classmethod
    def load(cls) -> "SearchState | None":
        """The state, or None when there is none yet."""
        if not STATE_FILE.is_file():
            return None
        try:
            return cls(**json.loads(STATE_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATE_FILE)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["round_number"] = len(self.rounds) + 1
        d["finished"] = len(self.rounds) >= self.max_rounds
        return d


def _clear_previous(base_version: str) -> Path | None:
    """Move a previous search's versions and episodes out of the way."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = RUNS_DIR.parent / "archive" / f"search_{stamp}"
    moved = 0

    for book in sorted(Policy.path_for("s0").parent.glob("cpt_lithology_rulebook_s*.md")):
        if not re.fullmatch(r"cpt_lithology_rulebook_s\d+\.md", book.name):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(book), str(dest / book.name))
        moved += 1

    for sounding in sorted(RUNS_DIR.glob("cpt_*")):
        for episode in sorted(sounding.iterdir()):
            if not episode.is_dir():
                continue
            if episode.name != base_version and not re.fullmatch(r"s\d+", episode.name):
                continue
            target = dest / sounding.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(episode), str(target / episode.name))
            moved += 1

    if STATE_FILE.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(STATE_FILE), str(dest / STATE_FILE.name))
        moved += 1

    return dest if moved else None


def start(cpt_ids: list[int], base_version: str = "v0", max_rounds: int = 15) -> SearchState:
    """Begin a search, archiving any previous one."""
    _clear_previous(base_version)
    state = SearchState(
        cpt_ids=[int(c) for c in cpt_ids], base_version=base_version,
        max_rounds=max_rounds, current_version=base_version,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    state.save()
    return state


def _round_version(state: SearchState, n: int) -> str:
    """Each round interprets under its own version name."""
    return state.base_version if n == 1 else f"s{n}"


def run_round(*, provider: str = DEFAULT_PROVIDER, model: str | None = None,
              pen: float = 5.0) -> dict:
    """Interpret every sounding once, and revise the practices if it erred."""
    state = SearchState.load()
    if state is None:
        raise RuntimeError("No search in progress. Start one first.")
    if len(state.rounds) >= state.max_rounds:
        raise RuntimeError(f"All {state.max_rounds} rounds have been run.")

    n = len(state.rounds) + 1
    version = _round_version(state, n)

    if version != state.current_version and not Policy.path_for(version).is_file():
        text = Policy.load(state.current_version).text.replace(
            f"Rulebook {state.current_version}", f"Rulebook {version}", 1)
        Policy.path_for(version).write_text(text, encoding="utf-8")


    trajectories = interpret_batch(state.cpt_ids, version, provider=provider,
                                   model=model, pen=pen)
    score = score_batch(state.cpt_ids, version)

    cards = {}
    for t in trajectories:
        cards[t.cpt_id] = {int(c["segment"]): c for c in t.load_state()}

    errors = []
    for cpt_id, segs in sorted(score.per_segment.items()):
        for g in segs:
            if g["correct"]:
                continue
            card = cards.get(cpt_id, {}).get(g["segment"], {})
            errors.append({
                "round": n,
                "rule_id": "+".join(r["id"] for r in state.active) or None,
                "cpt_id": cpt_id,
                "segment": g["segment"],
                "depth_label": g["depth_label"],
                "predicted_label": g["predicted_label"],
                "engineer_label": g["engineer_label"],
                "features": card.get("features", {}),
            })

    entry = {
        "round": n,
        "version": version,
        "rules": [r["id"] for r in state.active],
        "n_correct": score.n_correct,
        "n_segments": score.n_segments,
        "accuracy": score.accuracy,
        "metric_a": score.metric_a,
        "per_cpt": {str(c): r["n_correct"] for c, r in sorted(score.per_cpt.items())},
        "n_errors": len(errors),
        "rule_changed": False,
    }

    if not errors:
        state.rounds.append(entry)
        state.save()
        return entry


    state.memory.extend(errors)
    reflect_batch(trajectories, provider=provider, model=model)

    violations = constraint_violations(trajectories)
    stats = citation_stats(trajectories, score.per_segment, state.active, violations)
    for rule in state.active:
        s = stats.get(rule["id"], {})
        rule["cumulative_reward"] = rule.get("cumulative_reward", 0) + s.get("reward", 0)
    entry["citations"] = stats
    entry["violations"] = violations

    seeding = not state.active

    if seeding:
        result = call_provider(provider, SEED_SYSTEM,
                               build_seed_prompt(state.base_version, errors),
                               SEED_SCHEMA, model)
        state.active = [
            {"id": f"r{i + 1}", "round": n, "revisions": 0, "cumulative_reward": 0,
             "subsection": pr["subsection"], "text": pr["rule"],
             "rationale": pr["rationale"]}
            for i, pr in enumerate(result["practices"])
        ]
        changes = f"wrote {len(state.active)} practice(s) from the first round's errors"
    else:
        result = call_provider(
            provider, RULE_SYSTEM,
            build_rule_prompt(state.base_version, state.active, errors,
                              state.memory, state.retired, stats),
            RULE_SCHEMA, model,
        )
        by_id = {r["id"]: r for r in state.active}
        next_n = 1 + max([int(r["id"][1:]) for r in state.active + state.retired
                          if r["id"][1:].isdigit()] or [0])
        rebuilt = []

        for pr in result["practices"]:
            old_rule = by_id.pop(pr["id"], None)
            if old_rule is None:
                rebuilt.append({"id": f"r{next_n}", "round": n, "revisions": 0,
                                "cumulative_reward": 0, "subsection": pr["subsection"],
                                "text": pr["rule"], "rationale": pr["rationale"]})
                next_n += 1
            elif " ".join(pr["rule"].split()) == " ".join(old_rule["text"].split()):
                rebuilt.append(old_rule)
            else:
                state.retired.append(dict(old_rule))
                old_rule.update(subsection=pr["subsection"], text=pr["rule"],
                                rationale=pr["rationale"],
                                revisions=old_rule.get("revisions", 0) + 1,
                                cumulative_reward=0)
                rebuilt.append(old_rule)

        for dropped in by_id.values():
            state.retired.append(dict(dropped))

        state.active = rebuilt[:MAX_RULES]
        changes = result.get("changes", "")

    new_version = f"s{n + 1}"
    compose(state.base_version, state.active, new_version)
    state.current_version = new_version

    entry["rule_changed"] = True
    entry["seeding"] = seeding
    entry["changes"] = changes
    entry["rules_after"] = [dict(r) for r in state.active]
    state.rounds.append(entry)
    state.save()

    return entry
