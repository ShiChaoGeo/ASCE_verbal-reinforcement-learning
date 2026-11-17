# CPT Lithology Interpretation Rulebook v0

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


## Geotechnical knowledge

Every class is described on the same three lines: what it does on the Robertson
chart, how steady it is within a unit, and where it sits in the sequence.

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
