# Load Pattern, Flexibility, and DER Opportunity Analysis Engine
## Stage 1 + Stage 2 of 3 — Architecture/Ingestion/Daily Profiles + Analytical Engine

Implements **Implementation Handoff Specification v0.5**, Stage 1 and
Stage 2 scope (per the spec's own recommended staged delivery, Section 59):

> Stage 1: architecture, canonical schema, configuration parser,
> ingestion, validation, aggregation, time handling, daily profiles.
> Stage 2: analytical engine: features, peaks, valleys, shapes,
> clustering, pattern discovery, and meter coincidence.

Stage 3 (opportunity/scenario engine — shedding/shifting/modulation/
solar/storage, TOU, user-defined searches, visualization, full export
suite, final notebook assembly) is **not yet built**.

## Purpose

Analyze electricity demand data across one or many meters (individually,
in overlapping/hierarchical groups, and at portfolio level) to build a
canonical, gap-flagged, daily-profiled dataset that later stages use for
flexibility and DER opportunity analysis.

## Installation

```
pip install pandas numpy scipy scikit-learn matplotlib pytest --break-system-packages
```
Python 3.11+ required (uses stdlib `tomllib`).

## Required packages

pandas, numpy (Stage 1). scipy, scikit-learn, matplotlib are installed
for Stage 2/3 forward-compatibility but unused in Stage 1.

## Input requirements

A CSV (or future format) with columns mappable to the canonical fields:
`timestamp`, `meter_id`, `demand_kw`, and optionally `temperature_f`.
`demand_kw` is average demand over the interval; `timestamp` is
**interval-ending**. See `data/generate_synthetic_data.py` for the
shipped synthetic dataset and DATA_DICTIONARY.md for field definitions.

If the input file has no `temperature_f` column (or only a partial/
placeholder one), site temperature can instead be supplied from a
separate file via `data.external_temperature` in the config (see
`config/example_configuration.toml`). When `file_path` is set there,
it is always loaded and joined onto the observations by nearest
timestamp; by default it only fills rows missing `temperature_f`,
and `override_existing = true` makes it take precedence wherever a
match is found. `config/mcdonough_hall_configuration.toml` is a
worked example that overrides a meter export's placeholder
temperature column with a real weather-station series.

## Configuration

See `config/example_configuration.toml`. Controls: input file and
column mapping, time resolution, missing-data policy, meters and
(flat/overlapping/hierarchical) groups, portfolio composition, calendar
(holidays/seasons), and validation strictness. Sections beyond Stage 1
scope (`analysis.*`, `scenarios.*`, `time_periods`, `searches`) are
present as forward-compatible stubs; Stage 1 code does not read them.

## Notebook variants

- **`load_pattern_flexibility_der_analysis_consolidated.ipynb`** — the
  primary deliverable per Spec Section 1 ("single Jupyter/IPython
  notebook"). Fully self-contained: every function from `src/` is
  defined directly in the notebook, organized into sections. Runs with
  only `data/synthetic_load_data.csv` and
  `config/example_configuration.toml` alongside it — no `src/` import
  needed. Verified to execute clean in an isolated directory with no
  other files present. Ends with an inline **Tests** section (35
  assertions, run against the notebook's own live pipeline output, not
  synthetic fixtures) and the Stage 2 summary.
- **`load_pattern_flexibility_der_analysis.ipynb`** — the modular
  version that imports from `src/`, kept for anyone who wants to reuse
  the functions as an installable package or run `pytest tests/`
  separately. Both notebooks produce identical results.

## Notebook workflow

`load_pattern_flexibility_der_analysis.ipynb`:

1. Load + validate configuration (fails fast on structural errors).
2. Load input data, map to canonical schema, validate (negative/
   nonnumeric demand, duplicates, missing identifiers).
3. Detect native time resolution per meter.
4. Handle missing data (interpolate short gaps, leave large gaps
   flagged `missing`, never overwrite observed values).
5. Build calendar features (weekday/weekend/holiday, season).
6. Calculate interval energy (`demand_kw × interval_hours`).
7. Resolve meter groups (flat, overlapping, hierarchical) and portfolio.
8. Aggregate entity load by **summation** (never averaging) for every
   meter, group, and the portfolio.
9. Construct daily profiles (absolute + peak-normalized) and calculate
   daily features (mean/max/min, energy, load factor, peak-to-average,
   std, CV), flagging incomplete days.
10. Print Stage 1 summary.
11. Export native-resolution observations and daily features per entity
    to `output/`.

Stage 2 continues in the same notebook:

12. Time-of-day segment features (morning/midday/afternoon/evening
    peak, overnight/daytime mean) per entity.
13. Weekday-only change-point (balance-point) regression per meter
    (STATISTICAL; grid-searched breakpoint(s), least-squares
    baseload+slope): the full ASHRAE GL14 / IPMVP model family — 2P (no
    breakpoint), 3P heating, 3P cooling, 4P (shared breakpoint), 5P
    (independent heating+cooling breakpoints) — with automatic
    best-model selection via adjusted R².
14. Demand classification (threshold/percentile/rank — kept as separate
    flags, never collapsed into one generic "peak"), ramps, local
    peaks/valleys, and peak events (contiguous-with-gap-tolerance
    grouping) per entity.
15. Rule-based load-shape classification (independent flags +
    primary_shape) per entity/day.
16. K-means clustering of complete-day profiles, both absolute and
    peak-normalized, per meter and portfolio, with auto-k via
    silhouette score.
17. Pattern discovery: recurring peak timing, recurring shapes, and
    z-score outlier days.
18. Meter coincidence: peak-interval contribution %, diversity factor,
    interval-level coincidence rate.
19. Stage 2 summary and export.

## Output structure

```
output/
├── observations/native_resolution_observations.csv
├── daily/daily_features_<entity_id>.csv     (one per meter/group/portfolio)
├── peaks/peak_events_<entity_id>.csv
├── clusters/clusters_<entity_id>_<absolute|normalized>.csv
└── patterns/patterns_<entity_id>_<timing|shape|outliers>.csv
```

## Limitations (Stage 1 + 2)

- No opportunity/scenario engine, TOU, user-defined searches,
  visualization, or full export suite yet (Stage 3).
- Only CSV input is implemented (`file_format` extension point).
- `large_gap_method = "profile"` is a documented but unimplemented
  extension point; large gaps are left `missing`.
- Negative demand is unsupported by design (Spec Section 2.3);
  detected and rejected as an ERROR.
- Clustering: only K-means is implemented (Section 19 lists
  hierarchical/density-based/shape-distance as extension points).
  Auto-k via silhouette can behave poorly on near-flat, noisy profiles
  (observed on the synthetic B003 overnight-process meter, whose real
  variation is small relative to noise) — inspect silhouette scores,
  don't trust auto-k blindly for low-variance entities.
- Shape classification is rule-based/HEURISTIC (Section 18's required
  initial method); a statistical alternative using cluster assignments
  is a natural Stage 3+ extension.
- Change-point model: the full 2P/3P-heating/3P-cooling/4P/5P family is
  implemented (Section 13), with `select_best_change_point_model`
  fitting all five and picking the best by adjusted R² (which penalizes
  added parameters so a genuinely simpler relationship isn't overfit by
  the higher-parameter models). Multi-slope models (4P, 5P) constrain
  slopes non-negative via bounded least squares.

## Testing

`tests/test_stage1.py` (28 tests) + `tests/test_stage2.py` (28 tests) =
**56 tests**, all passing against the shipped synthetic dataset.

Stage 1: configuration validation, ingestion validation, time
resolution detection, energy calculation (15/30/60 min), calendar
features, missing-data interpolation and gap-size limits, meter group
resolution (flat/hierarchical/overlap), portfolio construction,
entity-load summation, daily profile/feature construction.

Stage 2: segment features, change-point model fit (2P/3P
heating/3P cooling/4P/5P, weekday-only, avoids weekday/weekend confound)
and its insufficient-data guard, best-model selection via adjusted R²,
temperature
banding, threshold/percentile/rank classification, ramp sign
correctness, local peak/valley detection, peak-event contiguous
grouping (including allowable-gap merging and the no-qualifying-interval
empty case), sustained-vs-short classification, shape classification
(including the insufficient-data path), clustering reproducibility
(fixed random_state), absolute-vs-normalized clustering both available,
cluster-size-sums-to-n-days invariant, pattern discovery (recurring
timing/shape/outliers — verified against the deliberately-injected
B002 unseasonal spike), and meter coincidence (contribution percentages
sum to ~100%, diversity factor ≥ 1 by construction, coincidence rate
bounds, and B003's overnight peaks correctly showing low coincidence
with B001's daytime peaks).

```
python3 -m pytest tests/ -v
```

## Extension points carried into code

- `data.input.file_format` — only "csv" implemented.
- `data.missing.large_gap_method` — only "leave_missing" implemented;
  "profile" is a stub.
- `analysis.profiles.normalization_method` — only "peak_normalized"
  implemented; min-max/z-score/mean/energy are stubs (Section 11).
- `analysis.clustering.method` — only K-means implemented.
- Change-point model — the full 2P/3P-heating/3P-cooling/4P/5P family
  and automatic best-model selection are implemented; curve-fit-based
  (rather than grid-search) breakpoint estimation and
  statistical-significance-based model selection (e.g. t-stat gating
  per ASHRAE GL14, rather than adjusted R² alone) are possible future
  refinements.
