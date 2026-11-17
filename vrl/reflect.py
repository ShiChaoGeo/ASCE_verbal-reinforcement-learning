"""Self-reflection as the verbal reward (Reflexion, Shinn et al., NeurIPS 2023)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from stratification_report import engineer_labels as reference_labels
from vrl.environment import ACTION_SPACE, OBSERVED_FEATURES
from vrl.providers import DEFAULT_PROVIDER, call_provider
from vrl.trajectory import Trajectory

FAILURE_MODES = (
    "rule_missing",
    "rule_too_vague",
    "wrong_feature_weighted",
    "sequence_ignored",
)

_SEGMENT_REF = re.compile(r"\b(?:segments?|seg\.?|units?)\s*#?\s*\d+", re.IGNORECASE)


def _reflection_schema(reveal_label: bool) -> dict:
    """Schema for one reflection pass."""
    properties = {
        "segment": {"type": "integer"},
        "failure_mode": {"type": "string", "enum": list(FAILURE_MODES)},
        "diagnosis": {
            "type": "string",
            "description": (
                "Why the rulebook, as applied, produced the wrong label here. "
                "Name the feature or the rule step that failed. This is audit "
                "text and MAY refer to this unit."
            ),
        },
        "proposed_rule": {
            "type": "string",
            "description": (
                "One interpretation practice that would prevent this class of "
                "error on a sounding you have never seen. Say whether it is "
                "geological — depositional origin and stratigraphic history — "
                "or geotechnical — CPTu response and its interpretation. Leave "
                "the class descriptions alone and propose no new section, "
                "procedure or checklist. A statement about evidence, not about "
                "this sounding: no unit numbers, no depths from this "
                "profile, no sounding name."
            ),
        },
        "generalizable": {
            "type": "boolean",
            "description": (
                "False when the diagnosis rests on something peculiar to this "
                "unit."
            ),
        },
    }
    required = ["segment", "failure_mode", "diagnosis", "proposed_rule", "generalizable"]

    if not reveal_label:
        properties["revised_label"] = {
            "type": "string",
            "enum": list(ACTION_SPACE),
            "description": (
                "Your second attempt at this unit, from re-reading the "
                "evidence. It must differ from the label you first assigned. "
                "Commit to what the evidence supports; do not cycle labels."
            ),
        }
        required.append("revised_label")

    return {
        "type": "object",
        "properties": {
            "reflections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "string",
                "description": (
                    "The ONE lesson this episode holds for the rulebook, stated "
                    "once, as a geological or geotechnical practice. Several "
                    "wrong units share a single cause far more often than "
                    "not; say what that cause is rather than repeating the list "
                    "above. Not a list of the units."
                ),
            },
        },
        "required": ["reflections", "summary"],
        "additionalProperties": False,
    }


_SHARED_RULES = f"""\
For every unit produce:

1. DIAGNOSIS — which step failed: a feature over-weighted, a rule that did not
   discriminate, context not used. "I was wrong" is not a diagnosis.

2. FAILURE MODE, exactly one of:
   rule_missing            nothing in the rulebook covers this situation
   rule_too_vague          a rule applies but does not discriminate here
   wrong_feature_weighted  right rule, wrong feature carried the decision
   sequence_ignored        context was on the cards and went unused

3. PROPOSED RULE — one interpretation practice, and whether it is geological
   (depositional origin and stratigraphic history) or geotechnical (CPTu
   response and its interpretation). Leave the class descriptions as they are
   and propose no new section, procedure or checklist. It is a statement about
   EVIDENCE, to be applied to soundings you have never seen: name no unit,
   no depth of this profile, no sounding. Thresholds on features are
   encouraged: which feature, which direction, what magnitude.

   A bullet marked [Hard constraint] is one every interpretation must satisfy.
   Never propose changing one; if an error broke it, propose the practice that
   would have kept it satisfied.

   Wrong units usually share one cause. Where they do, propose the SAME
   practice for each rather than a separate rule per unit.

     not a rule:  "Unit 6 is alluvial_clay."
     not a rule:  "Between 23.06 and 24.53 m the soil is alluvial clay."
     a rule:      "<label> rather than <label> when <feature> is <direction and
                   magnitude> relative to <a context described, not located>."

4. GENERALIZABLE — false if the diagnosis rests on something peculiar to this
   unit.

Cards carry: {', '.join(OBSERVED_FEATURES)}, plus depth range, point count and a
Robertson behaviour string. A rule that cannot be evaluated from those is
useless.
"""

_STAGE1_SYSTEM = f"""\
You are the self-reflection step of a verbal reinforcement learning loop for
CPTu lithology interpretation.

The policy labelled every unit of one sounding under a frozen rulebook and has
been scored. You are told WHICH units are wrong, not what they should be.

For each, commit to a REVISED LABEL that differs from the first. The policy
often names and then rejects the correct label — its stated reason for rejecting
one is frequently the step that failed.

{_SHARED_RULES}"""

_STAGE2_SYSTEM = f"""\
You are the self-reflection step of a verbal reinforcement learning loop for
CPTu lithology interpretation.

You already reflected on these units knowing only that they were wrong, and
your revised labels were still wrong. The engineer's labels are now disclosed.
The question is no longer what the label is, but what the rulebook lacks such
that the card did not lead you there — twice.

{_SHARED_RULES}"""


def _card_block(cards: list[dict], action: dict[int, dict], wrong: set[int]) -> str:
    """The full sounding as the reflector sees it, wrong units flagged."""
    parts = []
    for card in sorted(cards, key=lambda c: int(c["segment"])):
        seg = int(card["segment"])
        taken = action[seg]
        feats = ", ".join(
            f"{k}={v:.4g}"
            for k, v in (card.get("features") or {}).items()
            if isinstance(v, (int, float))
        )
        lines = [
            f"## Segment {seg} ({card['depth_label']}, n={card['n_points']})"
            + ("   <<< WRONG" if seg in wrong else "   (correct)"),
            f"- features: {feats}",
            f"- robertson: {card.get('robertson_behavior')}",
            f"- policy assigned: {taken['predicted_label']}",
            f"- policy reasoning: {taken.get('reasoning', '')}",
        ]
        if taken.get("evidence"):
            lines.append("- policy evidence: " + "; ".join(taken["evidence"]))
        if taken.get("rules_applied"):
            lines.append("- rules cited: " + "; ".join(taken["rules_applied"]))
        for alt in taken.get("rejected_alternatives") or []:
            lines.append(f"- policy REJECTED {alt['label']}: {alt['why_not']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _check_transferable(
    reflections: list[dict],
    cards: list[dict],
    cpt_id: int | str,
) -> list[str]:
    """Find proposed rules that encode this sounding instead of describing evidence."""
    depth_spans = {str(c["depth_label"]).replace(" m", "").strip() for c in cards}
    sounding_names = {f"cpt{cpt_id}", f"cpt {cpt_id}", f"cpt_{cpt_id}", f"cpt-{cpt_id}"}

    violations = []
    for item in reflections:
        rule = item.get("proposed_rule", "")
        seg = item.get("segment")
        if _SEGMENT_REF.search(rule):
            violations.append(
                f"unit {seg}: proposed_rule names a unit index — "
                f"{rule!r}"
            )
        for span in depth_spans:
            if span and span in rule:
                violations.append(
                    f"unit {seg}: proposed_rule quotes this profile's depth "
                    f"span {span!r} — {rule!r}"
                )
                break
        lowered = rule.lower()
        for name in sounding_names:
            if name in lowered:
                violations.append(
                    f"unit {seg}: proposed_rule names this sounding — {rule!r}"
                )
                break
    return violations


def _call_reflector(
    system: str,
    prompt: str,
    *,
    reveal_label: bool,
    cards: list[dict],
    cpt_id: int | str,
    provider: str,
    model: str | None,
    expected: set[int],
    predicted: dict[int, str] | None = None,
) -> dict:
    """One reflection pass, with a single retry when the output is not usable."""
    schema = _reflection_schema(reveal_label)
    feedback = ""

    for attempt in (1, 2):
        result = call_provider(provider, system, prompt + feedback, schema, model)
        reflections = result.get("reflections") or []

        problems = []
        got = {int(r["segment"]) for r in reflections}
        if got != expected:
            problems.append(
                f"Reflect on exactly these units: {sorted(expected)}. "
                f"You covered {sorted(got)}."
            )
        problems += _check_transferable(reflections, cards, cpt_id)
        if predicted is not None:
            for item in reflections:
                seg = int(item["segment"])
                if item.get("revised_label") == predicted.get(seg):
                    problems.append(
                        f"unit {seg}: revised_label repeats the label already "
                        f"known to be wrong ({predicted.get(seg)}). Commit to a "
                        f"different reading."
                    )

        if not problems:
            result["reflections"] = sorted(reflections, key=lambda r: int(r["segment"]))
            return result

        if attempt == 2:
            raise ValueError(
                "Reflection rejected twice:\n  - " + "\n  - ".join(problems)
            )
        feedback = (
            "\n\n# Your previous answer was rejected\n\nFix these and answer again:\n- "
            + "\n- ".join(problems)
        )

    raise AssertionError("unreachable")


def self_reflect(
    trajectory: Trajectory,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> dict:
    """Score the locked action, then reflect on it in two stages."""
    from vrl.policy import Policy

    cards = trajectory.load_state()
    action_list = trajectory.load_action()
    action = {int(a["segment"]): a for a in action_list}
    predicted = {seg: a["predicted_label"] for seg, a in action.items()}

    truth_list = reference_labels(int(trajectory.cpt_id))
    if len(truth_list) != len(cards):
        raise ValueError(
            f"CPT{trajectory.cpt_id}: {len(cards)} units but "
            f"{len(truth_list)} reference labels."
        )
    order = sorted(action)
    truth = dict(zip(order, truth_list))

    wrong = {seg for seg in order if predicted[seg] != truth[seg]}
    policy = Policy.load(trajectory.policy_version)

    print(
        f"\nSelf-reflection — CPT {trajectory.cpt_id} under policy "
        f"{trajectory.policy_version}"
    )
    print(
        f"Scored against engineer_label.csv: "
        f"{len(order) - len(wrong)}/{len(order)} correct."
    )

    stage1: dict = {"reflections": [], "summary": ""}
    stage2: dict = {"reflections": [], "summary": ""}
    self_corrected: set[int] = set()

    if wrong:
        base_prompt = (
            "# Rulebook the policy used\n\n"
            + policy.text
            + "\n\n# The sounding, and what the policy did with it\n\n"
            + _card_block(cards, action, wrong)
            + "\n\n# Scored outcome\n\n"
            + f"Wrong units: {sorted(wrong)}. "
            + f"Correct: {sorted(set(order) - wrong)}.\n"
            + "The correct labels are withheld.\n"
        )
        print(f"Stage 1 — binary feedback on {len(wrong)} unit(s)...")
        stage1 = _call_reflector(
            _STAGE1_SYSTEM,
            base_prompt + "\nReflect on every wrong unit.",
            reveal_label=False,
            cards=cards,
            cpt_id=trajectory.cpt_id,
            provider=provider,
            model=model,
            expected=wrong,
            predicted=predicted,
        )

        revised = {
            int(r["segment"]): r["revised_label"] for r in stage1["reflections"]
        }
        self_corrected = {s for s in wrong if revised.get(s) == truth[s]}
        still_wrong = wrong - self_corrected
        print(
            f"  self-corrected under binary feedback: "
            f"{len(self_corrected)}/{len(wrong)}"
        )

        if still_wrong:
            disclosure = "\n\n# Engineer labels, now disclosed\n\n" + "\n".join(
                f"- unit {s}: you assigned {predicted[s]}, "
                f"you revised to {revised.get(s)}, engineer says {truth[s]}"
                for s in sorted(still_wrong)
            )
            print(f"Stage 2 — labels revealed for {len(still_wrong)} unit(s)...")
            stage2 = _call_reflector(
                _STAGE2_SYSTEM,
                base_prompt + disclosure + "\nReflect on these units again.",
                reveal_label=True,
                cards=cards,
                cpt_id=trajectory.cpt_id,
                provider=provider,
                model=model,
                expected=still_wrong,
            )

    final = {int(r["segment"]): r for r in stage1["reflections"]}
    final.update({int(r["segment"]): r for r in stage2["reflections"]})
    stage_of = {int(r["segment"]): 1 for r in stage1["reflections"]}
    stage_of.update({int(r["segment"]): 2 for r in stage2["reflections"]})

    cards_by_seg = {int(c["segment"]): c for c in cards}
    records = []
    for seg in order:
        item = final.get(seg)
        is_correct = seg not in wrong
        records.append(
            {
                "segment": seg,
                "depth_label": cards_by_seg[seg]["depth_label"],
                "predicted_label": predicted[seg],
                "engineer_label": truth[seg],
                "is_correct": is_correct,
                "correction": None if is_correct else _correction_text(item),
                "generalizable": bool(item and item.get("generalizable")),
                "failure_mode": item["failure_mode"] if item else None,
                "reflection_stage": stage_of.get(seg),
                "self_corrected": seg in self_corrected,
                "revised_label": next(
                    (
                        r.get("revised_label")
                        for r in stage1["reflections"]
                        if int(r["segment"]) == seg
                    ),
                    None,
                ),
                "proposed_rule": item["proposed_rule"] if item else None,
            }
        )

    payload = {
        "cpt_id": trajectory.cpt_id,
        "policy_version_tested": trajectory.policy_version,
        "n_segments": len(records),
        "n_wrong": len(wrong),
        "segments": records,
        "reflexion": {
            "source": "self_reflection",
            "reference": "engineer_label.csv",
            "provider": provider,
            "model": model,
            "protocol": "two_stage_binary_then_disclosed",
            "n_self_corrected_stage1": len(self_corrected),
            "self_correction_rate": (
                len(self_corrected) / len(wrong) if wrong else None
            ),
            "stage1_summary": stage1["summary"],
            "stage2_summary": stage2["summary"],
        },
    }

    path = trajectory.save_verbal_reward(payload)
    doc = trajectory.save_reflection(_render_document(payload, action, policy.version))
    print(f"\nWrote {path}")
    print(f"Audit document: {doc}")

    return payload


def _correction_text(item: dict | None) -> str | None:
    """The verbal reward string the policy update consumes."""
    if not item:
        return None
    return (
        f"[{item['failure_mode']}] {item['diagnosis']}\n"
        f"Proposed rule: {item['proposed_rule']}"
    )


def _render_document(payload: dict, action: dict[int, dict], version: str) -> str:
    """The auditable record of one reflection."""
    meta = payload["reflexion"]
    segments = payload["segments"]
    wrong = [s for s in segments if not s["is_correct"]]
    n = payload["n_segments"]
    n_right = n - payload["n_wrong"]

    out = [
        f"# Reflection — CPT{payload['cpt_id']}, policy {version}",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Reflector: {meta['provider']}" + (f" / {meta['model']}" if meta["model"] else ""),
        f"- Reference: `{meta['reference']}`",
        f"- Accuracy_n (Metric B): {n_right}/{n} = {n_right / n:.3f}",
    ]
    if payload["n_wrong"]:
        out.append(
            f"- Self-corrected under binary feedback: "
            f"{meta['n_self_corrected_stage1']}/{payload['n_wrong']} "
            f"({meta['self_correction_rate']:.3f})"
        )
    out += [
        "",
        "## What the reflector was allowed to see",
        "",
        "The policy's action was locked before any of this ran. At no point did",
        "the reflector see the engineer's written reasoning, the geological",
        "cross-section, or the reference labels of any other sounding.",
        "",
        "| Stage | Disclosed | Withheld |",
        "| --- | --- | --- |",
        "| 1 | which units are wrong | the correct labels |",
        "| 2 | the correct labels, for units still wrong after stage 1 | — |",
        "",
        "Proposed rules were mechanically checked for unit indices, this",
        "profile's depth spans, and this sounding's name, and rejected if found:",
        "a rule that identifies the sounding is a cached answer, not a rule.",
        "",
        "## Outcome per unit",
        "",
        "| Seg | Depth | Policy | Engineer | Result | Stage | Failure mode |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in segments:
        if s["is_correct"]:
            result = "correct"
        elif s["self_corrected"]:
            result = "self-corrected"
        else:
            result = "needed disclosure"
        out.append(
            f"| {s['segment']} | {s['depth_label']} | `{s['predicted_label']}` | "
            f"`{s['engineer_label']}` | {result} | "
            f"{s['reflection_stage'] or '—'} | {s['failure_mode'] or '—'} |"
        )

    if wrong:
        out += ["", "## Diagnoses", ""]
        for s in wrong:
            taken = action[s["segment"]]
            out += [
                f"### Unit {s['segment']} ({s['depth_label']}) — {s['failure_mode']}",
                "",
                f"- Policy assigned `{s['predicted_label']}`; engineer says "
                f"`{s['engineer_label']}`.",
            ]
            if taken.get("rules_applied"):
                out.append("- Rules the policy cited: " + "; ".join(taken["rules_applied"]))
            for alt in taken.get("rejected_alternatives") or []:
                marker = "  **(this was the correct label)**" if alt["label"] == s["engineer_label"] else ""
                out.append(f"- Policy rejected `{alt['label']}`: {alt['why_not']}{marker}")
            if s["revised_label"]:
                out.append(
                    f"- Stage 1 revised label: `{s['revised_label']}` "
                    + ("— correct" if s["self_corrected"] else "— still wrong")
                )
            diagnosis = (s["correction"] or "").split("\nProposed rule: ")
            out += [
                "",
                f"**Diagnosis.** {diagnosis[0].split('] ', 1)[-1]}",
                "",
                f"**Proposed rule.** {s['proposed_rule']}",
                "",
                f"**Generalizable.** {s['generalizable']}",
                "",
            ]

        tally: dict[str, int] = {}
        for s in wrong:
            tally[s["failure_mode"]] = tally.get(s["failure_mode"], 0) + 1
        out += [
            "## Failure mode tally",
            "",
            "| Failure mode | Count |",
            "| --- | --- |",
        ]
        for mode, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            out.append(f"| `{mode}` | {count} |")
        out.append("")

        out += ["## Amendments carried into the policy update", ""]
        for s in wrong:
            if s["generalizable"]:
                out.append(f"- {s['proposed_rule']}")
        withheld = [s for s in wrong if not s["generalizable"]]
        if withheld:
            out += [
                "",
                "Not carried forward (marked non-generalizable):",
                "",
            ]
            for s in withheld:
                out.append(
                    f"- unit {s['segment']} ({s['failure_mode']}): "
                    f"{s['proposed_rule']}"
                )
        out.append("")

    out += ["## Reflector's summary", ""]
    if meta["stage1_summary"]:
        out += ["**Stage 1.** " + meta["stage1_summary"], ""]
    if meta["stage2_summary"]:
        out += ["**Stage 2.** " + meta["stage2_summary"], ""]
    if not payload["n_wrong"]:
        out += ["No unit was wrong.", ""]

    return "\n".join(out)
