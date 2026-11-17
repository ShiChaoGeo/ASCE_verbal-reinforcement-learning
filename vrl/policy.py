"""Policy: a rulebook in natural language, executed by an LLM."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from vrl.environment import ACTION_SPACE, State
from vrl.providers import DEFAULT_PROVIDER, call_provider

DATA_DIR = Path(__file__).resolve().parent.parent

RULEBOOK_DIR = DATA_DIR / "rulebooks"


@dataclass(frozen=True)
class Policy:
    """Rulebook v_k."""

    version: str
    text: str

    @classmethod
    def load(cls, version: str) -> "Policy":
        path = cls.path_for(version)
        if not path.is_file():
            raise FileNotFoundError(f"Missing rulebook {path}")
        return cls(version=version, text=path.read_text(encoding="utf-8"))

    @staticmethod
    def path_for(version: str) -> Path:
        return RULEBOOK_DIR / f"cpt_lithology_rulebook_{version}.md"

    def __repr__(self) -> str:
        return f"Policy({self.version}, {len(self.text)} chars)"


def action_schema(n_segments: int) -> dict:
    """Schema for a full-sounding action."""
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "array",
                "minItems": n_segments,
                "maxItems": n_segments,
                "items": {
                    "type": "object",
                    "properties": {
                        "segment": {"type": "integer"},
                        "predicted_label": {"type": "string", "enum": list(ACTION_SPACE)},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Feature values that drove the choice, quoted with "
                                "their numbers, e.g. 'Ic_median = 3.42 -> clay-like'. "
                                "Do not invent values."
                            ),
                        },
                        "rules_applied": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Rulebook rules used, quoted or named.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Why THIS label for THIS unit: how the evidence "
                                "combines with the rules, plus depth context."
                            ),
                        },
                        "rejected_alternatives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "enum": list(ACTION_SPACE)},
                                    "why_not": {"type": "string"},
                                },
                                "required": ["label", "why_not"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "segment", "predicted_label", "evidence",
                        "rules_applied", "reasoning", "rejected_alternatives",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are interpreting CPTu lithology for a geotechnical study.

    You are given a frozen rulebook and feature cards for every unit of one
    sounding, shallow to deep.

    Read the whole sounding before you label any of it. The cards are in depth
    order, but that is not the order in which the sounding is understood: a
    unit near the base can settle what a unit near the surface is. Compare
    across the entire profile first; what that comparison settles binds every
    label that follows. Only then assign the labels, in a single pass; nothing
    downstream revises them.

    A rulebook bullet marked [Hard constraint] is not guidance, and not a
    preference to be weighed against the evidence. It ADMITS NO EXCEPTION. An
    interpretation that breaks one is not a judgement call; it is invalid,
    however well the reading is argued.

    Two of them can never genuinely conflict, because a valid interpretation
    always exists. If satisfying one appears to force breaking another, the
    error is earlier: a boundary you have already placed sits in the wrong
    unit. Go back and revise it.

    Before you commit, read your labels top to bottom against every [Hard
    constraint] in the rulebook, and revise until none of them fails.

    Rules of this task:
    - Use only the rulebook and the cards: no raw curves, no engineer labels,
      no site history. Quote card numbers as evidence; never invent a value.
    - Units are in depth order. What lies above and below a unit is evidence
      about what it is.
    - When two labels are close, name the one you rejected and what ruled it
      out.

    A geotechnical reviewer must be able to follow, from the card numbers and
    the rulebook text, why each unit got its label.
    """
)


def build_prompt(policy: Policy, cards: list[dict]) -> str:
    """Assemble the observation the policy conditions on."""
    parts = ["# Rulebook (frozen — this is the policy under test)\n", policy.text]
    parts += [
        "\n\n# Unit feature cards for the CPT to interpret\n",
        json.dumps(cards, indent=2, ensure_ascii=False),
        f"\n\nAssign a label to all {len(cards)} units.",
    ]
    return "".join(parts)


def validate_action(result: dict, cards: list[dict]) -> list[dict]:
    """Reject a malformed action while the engineer answer is still unseen."""
    action = result.get("action")
    if not isinstance(action, list):
        raise ValueError("Response has no 'action' list.")

    expected = {int(c["segment"]) for c in cards}
    got = {int(a["segment"]) for a in action}
    if got != expected:
        raise ValueError(
            f"Unit coverage mismatch. Missing: {sorted(expected - got)}; "
            f"unexpected: {sorted(got - expected)}"
        )

    bad = {a["predicted_label"] for a in action} - set(ACTION_SPACE)
    if bad:
        raise ValueError(f"Labels outside ACTION_SPACE: {sorted(bad)}")

    for a in action:
        if not a.get("reasoning", "").strip():
            raise ValueError(f"Unit {a['segment']} has empty reasoning.")
        if not a.get("evidence"):
            raise ValueError(f"Unit {a['segment']} cites no evidence.")

    return sorted(action, key=lambda a: int(a["segment"]))


PHASE_OF = {"fill": 1, "disturbed_marine_deposit": 1, "marine_deposit": 2,
            "alluvial_sand": 3, "alluvial_clay": 3}

BEHAVIOUR_SWAP = {
    1: ("disturbed_marine_deposit", "fill"),
    3: ("alluvial_clay", "alluvial_sand"),
}

MAX_RETRIES = 3


class HardConstraintUnmet(Exception):
    """Raised when no attempt produced an interpretation that satisfies them."""


def hard_constraint_violations(action: list[dict], cards: list[dict]) -> list[dict]:
    """Where an interpretation breaks a [Hard constraint], in unit order."""
    hits = []
    phases = [(int(a["segment"]), a["predicted_label"], PHASE_OF[a["predicted_label"]])
              for a in action]
    for (u0, l0, p0), (u1, l1, p1) in zip(phases, phases[1:]):
        if p1 < p0:
            hits.append({"unit": u1, "constraint": "phase order",
                         "detail": (f"unit {u0} is {l0} and unit {u1} is {l1}: the "
                                    "profile goes back to an earlier phase")})

    alluvial = [(u, l) for u, l, ph in phases if ph == 3]
    missing = {"alluvial_sand", "alluvial_clay"} - {l for _, l in alluvial}
    if missing:
        where = (f"the alluvial interval runs from unit {alluvial[0][0]} to unit "
                 f"{alluvial[-1][0]} and is {alluvial[0][1]} throughout"
                 if alluvial else
                 "the profile has no alluvial interval at all")
        hits.append({
            "unit": alluvial[0][0] if alluvial else phases[-1][0],
            "constraint": "alluvial pair",
            "detail": (f"{where}: {' and '.join(sorted(missing))} "
                       f"{'do' if len(missing) > 1 else 'does'} not appear at all"),
        })
    return hits


SAND_BAND = "sand-like"
SANDY_SILT_BAND = "between sand-like and clay-like, closer to sandy / sandy-silty"
SILT_BAND = "between sand-like and clay-like / silty"
CLAY_BAND = "clay-like"

ADMISSIBLE_BANDS = {
    "fill": {SAND_BAND, SANDY_SILT_BAND, SILT_BAND},
    "disturbed_marine_deposit": {CLAY_BAND},
    "marine_deposit": {CLAY_BAND},
    "alluvial_sand": {SAND_BAND, SANDY_SILT_BAND, SILT_BAND},
    "alluvial_clay": {CLAY_BAND, SILT_BAND},
}


def behaviour_violations(labels: list[str], cards: list[dict]) -> list[dict]:
    """Units whose label is outside the Robertson bands the rulebook allows it."""
    out = []
    for i, (label, card) in enumerate(zip(labels, cards)):
        band = card.get("robertson_behavior", "")
        allowed = ADMISSIBLE_BANDS.get(label)
        if allowed is None or not band or band == "unknown":
            continue
        if band not in allowed:
            out.append({"unit": i + 1, "label": label, "behaviour": band,
                        "detail": f"unit {i + 1} is {band}, which "
                                  f"{label} does not occupy"})
    return out


def _repair_prompt(policy: "Policy", cards: list[dict], labels: list[str],
                   hits: list[dict], first: int) -> str:
    """Send the offending units back, with the ground below them open too."""
    profile = "\n".join(
        f"  unit {i + 1}: {lab}   ({cards[i].get('depth_label', '')}, "
        f"{cards[i].get('robertson_behavior', '')})"
        for i, lab in enumerate(labels))
    named = "\n".join(f"  unit {h['unit']}: you said {h['label']}. {h['detail']}."
                       for h in hits)
    return "".join([
        "# Rulebook (frozen \u2014 this is the policy under test)\n", policy.text,
        "\n\n# Unit feature cards for the CPT to interpret\n",
        json.dumps(cards, indent=2, ensure_ascii=False),
        "\n\n# Your interpretation of this sounding, as it stands\n", profile,
        "\n\n# Units sent back to you\n",
        "These units break a [Hard constraint] that admits no exception: a "
        "clay class sits on a clay-like unit and a sand class does not.\n\n",
        named,
        f"\n\nRe-label them. You are NOT tied to the phase each one is in now: "
        f"if the fault is that a boundary sits in the wrong unit, move the unit "
        f"across it. Units {first} to {len(cards)} are open to you for that "
        f"reason, and units 1 to {first - 1} are not -- nothing above the "
        f"shallowest breach was in question.\n\n"
        "Whatever you return must leave the whole sounding satisfying all "
        "three constraints together:\n"
        "  1. the phase order runs reclamation (fill, disturbed_marine_deposit) "
        "-> marine (marine_deposit) -> alluvial (alluvial_sand, alluvial_clay) "
        "and never goes back;\n"
        "  2. the alluvial interval exists and carries both of its soils;\n"
        "  3. every label agrees with its unit's Robertson behaviour.\n\n"
        "Return every unit you change, including any you have to change only "
        "to keep the order intact.",
    ])


def repair_schema(units: list[int]) -> dict:
    """Schema for a repair: any of the open units, any label."""
    schema = action_schema(len(units))
    item = schema["properties"]["action"]["items"]["properties"]
    item["segment"] = {"type": "integer", "enum": list(units)}
    schema["properties"]["action"]["minItems"] = 1
    schema["properties"]["action"]["maxItems"] = len(units)
    return schema


def _settled(labels: list[str], cards: list[dict]) -> list[dict]:
    """Every breach left in a profile, of all three constraints."""
    action = [{"segment": i + 1, "predicted_label": l}
              for i, l in enumerate(labels)]
    return hard_constraint_violations(action, cards) + behaviour_violations(labels, cards)


def correct_behaviour(labels: list[str], cards: list[dict], policy=None,
                      provider: str = DEFAULT_PROVIDER, model: str | None = None,
                      attempts: int = 2) -> tuple[list[str], list[dict]]:
    """Re-label the offending units, leaving the sounding above them alone."""
    fixed, changes = list(labels), []
    hits = behaviour_violations(fixed, cards)
    if not hits:
        return fixed, changes

    for _ in range(attempts if policy is not None else 0):
        first = min(h["unit"] for h in hits)
        units = list(range(first, len(cards) + 1))
        try:
            reply = call_provider(provider, SYSTEM_PROMPT,
                                  _repair_prompt(policy, cards, fixed, hits, first),
                                  repair_schema(units), model)
        except Exception as exc:
            print(f"  ! behaviour repair call failed ({exc})")
            break
        trial = list(fixed)
        for item in reply.get("action", []):
            i = int(item["segment"]) - 1
            if i >= first - 1 and item["predicted_label"] in ACTION_SPACE:
                trial[i] = item["predicted_label"]
        if not _settled(trial, cards):
            fixed = trial
            break
        hits = behaviour_violations(trial, cards) or hits
        fixed = trial if behaviour_violations(trial, cards) != hits else fixed

    if policy is None:
        for hit in behaviour_violations(fixed, cards):
            i = hit["unit"] - 1
            pair = BEHAVIOUR_SWAP.get(PHASE_OF[fixed[i]])
            if pair is None:
                continue
            want = pair[0] if hit["behaviour"] == "clay-like" else pair[1]
            if want != fixed[i]:
                fixed[i] = want

    if _settled(fixed, cards):
        return list(labels), []

    for i, (was, now) in enumerate(zip(labels, fixed)):
        if was != now:
            changes.append({"unit": i + 1, "label": was, "corrected_to": now,
                            "behaviour": cards[i].get("robertson_behavior", ""),
                            "detail": f"unit {i + 1} was {was}",
                            "outcome": f"re-labelled {now}"})
    return fixed, changes


def _retry_note(hits: list[dict]) -> str:
    """What to tell the interpreter about its own breaches, without the answer."""
    lines = ["\n\n# Your previous interpretation of this sounding was rejected\n",
             "It broke a [Hard constraint], which admits no exception:\n"]
    lines += [f"- {h['detail']}\n" for h in hits]
    lines.append(
        "\nDo not patch the units named above one by one. A breach of this kind "
        "means a boundary you placed sits in the wrong unit, so the units around "
        "it cannot be labelled without contradiction. Find that boundary, move "
        "it, and label the sounding again from the top.\n"
    )
    return "".join(lines)


def act(
    policy: Policy,
    state: State | list[dict],
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> list[dict]:
    """Run the policy on a state and return one validated action per unit."""
    cards = state.cards if isinstance(state, State) else state
    prompt = build_prompt(policy, cards)
    schema = action_schema(len(cards))

    for attempt in range(MAX_RETRIES + 1):
        action = validate_action(
            call_provider(provider, SYSTEM_PROMPT, prompt, schema, model), cards)
        hits = hard_constraint_violations(action, cards)
        if not hits:
            labels = [a["predicted_label"] for a in action]
            fixed, changes = correct_behaviour(labels, cards, policy,
                                               provider=provider, model=model)
            for item, was, now in zip(action, labels, fixed):
                if was != now:
                    item["predicted_label"] = now
                    item["behaviour_corrected_from"] = was

            hits = (hard_constraint_violations(action, cards)
                    + behaviour_violations(
                        [a["predicted_label"] for a in action], cards))
            if not hits:
                for ch in changes:
                    print(f"  ~ behaviour: {ch['detail']}; {ch['outcome']}")
                return action
            print("  ! the repair could not settle the sounding; re-interpreting")
        if attempt == MAX_RETRIES:
            raise HardConstraintUnmet(
                f"{MAX_RETRIES + 1} attempts all broke a hard constraint: "
                + "; ".join(h["detail"] for h in hits)
            )
        print(f"  ! hard constraint broken ({len(hits)}); re-interpreting "
              f"(attempt {attempt + 2} of {MAX_RETRIES + 1})")
        prompt = build_prompt(policy, cards) + _retry_note(hits)
