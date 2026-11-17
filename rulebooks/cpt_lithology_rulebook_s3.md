# CPT Lithology Interpretation Rulebook s3

Assigns a lithological label to each change-point-detected unit of a CPT/CPTu sounding at a Hong Kong reclamation site.

## Target Labels

1. `fill`
2. `disturbed_marine_deposit`
3. `marine_deposit`
4. `alluvial_sand`
5. `alluvial_clay`

## Interpretation Assumptions

- Change point detection has already divided each CPT profile into units，Consecutive units may legitimately carry the same label.
- [Hard constraint] Judge the soil behaviour of each unit using `Ic` and the Robertson chart first； then, according to the soil behavior description in `Geotechnical knowledge` below, to choose the matched soil types as candidates.


## Geological knowledge

- Despotional sequence: marine (`marine_deposit`) is the quietest, cleanest clay-like block in the profile; 
- [Hard constraint] all units above  (`marine_deposit`) is reclamation (`fill` and `disturbed_marine_deposit`), and all units below  (`marine_deposit`) is alluvial (`alluvial_sand` and `alluvial_clay`).
- [r1] When a laterally plausible marine clay package is recognised by several consecutive clay-like units that are collectively the quietest part of the profile and show sustained positive pore-pressure response, treat that whole package as the marine datum only while it remains the deepest such package before the profile passes into the alluvial interval; once clay-like or sand-like units below it show persistent alluvial-style alternation or stronger within-unit variability, classify those deeper units as alluvial rather than extending the marine package downward.

## Geotechnical knowledge

Every class is described on the same three lines: what it does on the Robertson
chart, how steady it is within a unit, and where it sits in the sequence.
- [r3] For sand-like units, do not use sand-like Robertson behaviour alone to infer alluvial_sand: if the unit lies in the reclamation interval above the marine datum and shows heterogeneous friction/cone response or alternation with disturbed clay-like reclamation units, classify it as fill even when Ic is cleanly sand-like and Bq is near zero.
- [r4] Within the alluvial interval below the marine datum, classify a unit as alluvial_sand rather than alluvial_clay when its Ic is only marginally clay-like or sand-like and its pore-pressure response is near zero or negative and steady, especially if qc and fs variability are not strongly clay-like.

### 1. Fill

- **Behaviour** — mainly sand-like; `Ic` may fall between sand-like and clay-like on the Robertson chart.
- **Variability** — obvious, repeated oscillation in `Ic` and cone/friction measurements; the amplitude is not necessarily large.
- **Sequence** — reclamation phase, above the marine deposit; may be interbedded with `disturbed_marine_deposit`.

### 2. Disturbed Marine Deposit

- **Behaviour** — mainly clay-like.
- **Variability** — clearly more oscillation in `Ic` than the undisturbed marine deposit.
- **Sequence** — reclamation phase, above the marine deposit; occurs near or interbedded with `fill`.

### 3. Marine Deposit

- **Behaviour** — clean clay-like on the Robertson chart; usually excess pore-water pressure.
- **Variability** — the lowest in the profile: low noise in `Ic`, `qc` and `fs`.
- **Sequence** — the datum for the profile, below `fill` or `disturbed_marine_deposit`，and above alluvial soil below.

### 4. Alluvial Sand

- **Behaviour** — sandy-silty on the Robertson chart, mainly sand-like.
- **Variability** — strong: high dispersion in `Ic`, `qc` and `fs`.
- **Sequence** — alluvial phase, below the marine deposit; commonly interbedded with `alluvial_clay`.

### 5. Alluvial Clay

- **Behaviour** — clayey-silty on the Robertson chart, mainly clay-like.
- **Variability** — strong: high dispersion in `Ic`, `qc` and `fs`.
- **Sequence** — alluvial phase, below the marine deposit; commonly interbedded with `alluvial_sand`.
