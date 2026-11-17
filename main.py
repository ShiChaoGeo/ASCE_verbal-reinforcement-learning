"""CLI for the Verbal RL CPT pipeline."""

from __future__ import annotations

import argparse
import re

from vrl.config import status as key_status
from vrl.environment import (
    MIN_THICKNESS_M,
    available_application_keys,
    available_cpt_ids,
)
from vrl.policy import RULEBOOK_DIR
from vrl.providers import DEFAULT_PROVIDER, PROVIDER_NAMES


def _rulebook_versions() -> list[str]:
    """Rulebook versions on disk: v0 first, then the search rounds in order."""
    found = []
    for path in RULEBOOK_DIR.glob("cpt_lithology_rulebook_*.md"):
        m = re.fullmatch(r"cpt_lithology_rulebook_([vs])(\d+)\.md", path.name)
        if m:
            found.append((m.group(1) != "v", int(m.group(2)), m.group(1) + m.group(2)))
    return [name for *_, name in sorted(found)]


def _status() -> str:
    train_ids = available_cpt_ids()
    app_keys = available_application_keys()
    versions = _rulebook_versions()
    return "\n".join([
        f"Training CPTs (cpt_i.xlsx): "
        f"{', '.join(str(i) for i in train_ids) if train_ids else '(none)'}",
        f"Application CPTs: {', '.join(app_keys) if app_keys else '(none)'}",
        f"Rulebooks: {', '.join(versions) if versions else '(none)'}",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verbal RL for CPT lithology interpretation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("keys", help="Show which provider API keys are configured")
    sub.add_parser("status", help="Show the soundings and rulebooks on disk")

    p_start = sub.add_parser("search-start", help="Begin a new rulebook search")
    p_start.add_argument("--cpts", default=None,
                         help="Training soundings, comma separated (default: all)")
    p_start.add_argument("--rounds", type=int, default=15,
                         help="Maximum number of rounds (default 15)")

    p_round = sub.add_parser(
        "search-round",
        help="Run one round of the rulebook search",
    )
    p_round.add_argument("--provider", default=DEFAULT_PROVIDER, choices=PROVIDER_NAMES)
    p_round.add_argument("--model", default=None)
    p_round.add_argument("--pen", type=float, default=5.0)

    p_apply = sub.add_parser(
        "interpret",
        help="Apply a rulebook to a folder of unlabelled soundings",
    )
    p_apply.add_argument("--folder", default=None,
                         help="Folder of soundings, e.g. case_study_1")
    p_apply.add_argument("--policy", default=None, help="Rulebook version, e.g. s10")
    p_apply.add_argument("--cpt", default=None,
                         help="Soundings to interpret, comma separated (default: all)")
    p_apply.add_argument("--draws", type=int, default=3,
                         help="Interpretations per sounding (default 3)")
    p_apply.add_argument("--add", action="store_true",
                         help="Add --draws draws to those already on disk")
    p_apply.add_argument("--resume", action="store_true",
                         help="Continue the unfinished interpretation")
    p_apply.add_argument("--provider", default=DEFAULT_PROVIDER, choices=PROVIDER_NAMES)
    p_apply.add_argument("--model", default=None)
    p_apply.add_argument("--pen", type=float, default=5.0)
    p_apply.add_argument("--min-thickness", type=float, default=None,
                         help="Minimum unit thickness in metres (default 1/6)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "keys":
        print(key_status())

    elif args.command == "status":
        print(_status())

    elif args.command == "search-start":
        from vrl.search import start
        cpts = ([int(c) for c in args.cpts.split(",") if c.strip()] if args.cpts
                else available_cpt_ids())
        state = start(cpts, max_rounds=args.rounds)
        print(f"search started on CPT {state.cpt_ids}, up to {state.max_rounds} rounds")

    elif args.command == "search-round":
        from vrl.search import run_round
        entry = run_round(provider=args.provider, model=args.model, pen=args.pen)
        print(f"round {entry['round']}: {entry['n_correct']}/{entry['n_segments']} "
              f"= {entry['accuracy']:.3f}   rule_changed={entry['rule_changed']}")

    elif args.command == "interpret":
        from vrl import interpret as apply_phase
        if not args.resume:
            if not (args.folder and args.policy):
                raise SystemExit("--folder and --policy are required unless --resume")
            only = None
            if args.cpt:
                from vrl.environment import normalize_application_key
                only = [normalize_application_key(c.strip())
                        for c in args.cpt.split(",") if c.strip()]
            apply_phase.start(args.folder, args.policy, args.draws, only,
                              append=args.add)
        summary = apply_phase.run(
            provider=args.provider, model=args.model, pen=args.pen,
            min_thickness=(MIN_THICKNESS_M if args.min_thickness is None
                           else args.min_thickness))
        for key in summary["done"]:
            r = summary["results"][key]
            print(f"  {key}: {r['n_segments']} units, {r['draws_used']} draws, "
                  f"mean agreement {r['agreement']:.2f}, "
                  f"{r['unanimous']}/{r['n_segments']} unanimous")
        print(f"interpreted {summary['n_done']}/{summary['n_total']} soundings "
              f"in {summary['folder']} under {summary['policy_version']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
