# CPTu Lithology Interpretation through Few-shot Verbal Reinforcement Learning

A large language model interprets the lithology of cone penetration tests
(CPTu) by following a natural-language **rulebook**. Learning changes the
rulebook, never the model weights, so every version is a readable file that an
engineer can audit.

This folder holds the code and the rulebooks. No dataset is included; the
expected layout of the data is described under [Data](#data).

---

## Framework

```text
 engineer's knowledge (text)   CPTu records (qc, fs, u2 -> Ic, Bq)
 Robertson chart behaviour     stratigraphic sequence
                │  fused as text: rulebook + per-segment feature cards
                ▼
 Rulebook v0 (fixed)  +  at most three learned practices
                │
                ▼
 ┌─ Training: verbal RL on the labelled soundings ────────────────────┐
 │  interpret + lock  ->  scalar reward (accuracy)                    │
 │        ▲                    │                                      │
 │        │                    ▼                                      │
 │  revise the practices  <-  verbal reward (reflection on errors)    │
 └────────────────────────────────────────────────────────────────────┘
                │  frozen rulebook
                ▼
 Application: repeated interpretation of unlabelled soundings
              -> most probable profile + semantic entropy U_s per unit
```

| RL element | In this code |
| --- | --- |
| Environment | one CPTu sounding, segmented by PELT into units |
| State | per-segment feature cards and the current rulebook |
| Action | a lithology label for every segment, locked at once |
| Policy | the rulebook, executed by a fixed LLM |
| Scalar reward | accuracy of the locked action against the engineer's labels |
| Verbal reward | the policy's written diagnosis of each error |
| Policy update | keep, reword, drop or add a learned practice |

### Training: the one-rule search

Each round interprets every training sounding with `v0` plus the current
practices, locks the labels, scores them, reflects on every wrong unit and
revises the practices.

- `vrl.search.compose` copies `v0` byte for byte and appends the practices as
  `- [rN] ...` bullets, so the engineer's rules and `[Hard constraint]` lines
  cannot be edited by the search.
- A practice is credited only where it was cited:
  `R(p) = right - 2 * (wrong + breaches)`.
- Every error is kept with the round and the practices in force, so the
  reflection can tell a practice that does not reach a unit from one that
  overreached.
- Rounds are started by hand, after the practices and the learning curve have
  been read.

### Application

A finished rulebook interprets each unlabelled sounding several times. The
draws are reduced to the admissible profile with the most votes; each unit
carries its probability and the semantic entropy `U_s` of its votes.

Every locked action must keep the phase order (reclamation, marine, alluvial,
never reversed with depth) and keep each label within the Robertson behaviour
bands the rulebook allows it (`vrl.policy.ADMISSIBLE_BANDS`).

### Accuracy

The accuracy reported for an interpreted sounding is depth weighted (Metric A):
the metres whose label agrees with the engineer's section, divided by the
metres the sounding measured. Ground the sounding did not reach is left out.

---

## Layout

```text
main.py                CLI
vrl/
  environment.py       soundings, segmentation, feature cards, action space
  policy.py            rulebook loading, the policy call, hard constraints
  batch.py             interpreting and scoring a batch of training soundings
  reflect.py           self-reflection on wrong units
  reward.py            scalar reward
  search.py            the one-rule search over learned practices
  interpret.py         application to unlabelled soundings, consensus, U_s
  trajectory.py        on-disk record of one interpreted sounding
  providers.py         LLM providers
  config.py            API keys
cpt_segmentation.py    loading a CPTu record, Ic, PELT segmentation
segment_features.py    per-segment features
ground_truth.py        the engineer's cross-section of the training site
stratification_report.py   engineer labels mapped onto the segments
evaluate_cpt.py        Metric A and Metric B for one training sounding
rulebooks/             rulebook v0 and versions s2 to s10 of one search
```

### Rulebooks

`rulebooks/` holds the ten versions of the search shown in the recording:
`v0`, the engineer's rulebook, and `s2` to `s10`, one per round. Each later
version is `v0` with the learned practices of that round appended.

| Version | Accuracy (Metric B) on the training soundings |
| --- | --- |
| v0 | 0.833 |
| s2 | 0.848 |
| s3 | 0.955 |
| s4 | 0.970 |
| s5 | 0.985 |
| s6 to s10 | 1.000 |

---

## Data

The data are not distributed with the code. The code expects them beside
`main.py`:

```text
cpt_1.xlsx ... cpt_4.xlsx     labelled training soundings
engineer_label.csv            cpt_id, segment, soil_id, label: one row per segment
ground_truth.npy              the training site's section, class ids
case_study_*/                 one folder per application site
  cpt_unknown_*.xlsx          unlabelled soundings
  ground_truth.npy            the site's section, class ids (optional, for scoring)
  section.json                {"resolution": 6, "columns": {"unknown_1": 7, ...}}
```

`section.json` gives the vertical resolution of the section in rows per metre
and the column each sounding stands in. Class ids of an application section
follow `vrl.environment.ACTION_SPACE`; those of the training section follow
`ground_truth.ID_TO_LABEL`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # fill in at least one provider key
python main.py keys          # which keys are configured
python main.py status        # soundings and rulebooks found
```

Providers: OpenAI (`OPENAI_API_KEY`, default), Anthropic
(`ANTHROPIC_API_KEY`), Gemini (`GEMINI_API_KEY`), DeepSeek
(`DEEPSEEK_API_KEY`).

## Usage

Search: begin it once, then run one round at a time.

```bash
python main.py search-start --cpts 1,2,3,4 --rounds 15
python main.py search-round --provider openai
```

A new search moves the rulebooks and episodes of the previous one to
`archive/search_<stamp>/`.

The search state, the error memory and the reward of each practice are kept in
`vrl_runs/search/state.json`.

Application:

```bash
python main.py interpret --folder case_study_1 --policy s10 --draws 10
python main.py interpret --folder case_study_1 --policy s10 --cpt unknown_3 --draws 5 --add
python main.py interpret --resume
```

Results are written to `vrl_runs/apply/{folder}/{rulebook}/{sounding}.json`.
