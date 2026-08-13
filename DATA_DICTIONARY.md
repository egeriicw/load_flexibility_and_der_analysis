# Data Dictionary — Stage 1

## Canonical input fields

| Field | Type | Required | Unit | Meaning | Source/Derived |
|---|---|---|---|---|---|
| timestamp | datetime | Yes | — | Interval-ending timestamp | Source (mapped) |
| meter_id | string | Yes | — | Physical meter identifier | Source (mapped) |
| demand_kw | float | Yes | kW | Average demand over the interval | Source (mapped); superseded downstream by observed/interpolated/analysis_demand_kw |
| temperature_f | float | No | °F | Site/area weather-station temperature | Source (mapped) |

## Derived fields (Stage 1)

| Field | Type | Unit | Meaning | Calculation | Missing-data behavior |
|---|---|---|---|---|---|
| observed_demand_kw | float | kW | Original demand value, untouched | = raw demand_kw at that interval | NaN if interval absent from source |
| interpolated_demand_kw | float | kW | Value filled by interpolation | pandas linear/time interpolation, limited by max_interpolation_intervals | NaN where observed, NaN if gap too large |
| analysis_demand_kw | float | kW | Value used by all downstream analysis | observed if present, else interpolated_demand_kw | NaN if neither available (large gap) |
| is_observed | bool | — | True if the interval had a source value | demand_kw notna at reindex | — |
| is_interpolated | bool | — | True if value was filled by interpolation | not is_observed AND interpolated value present | — |
| data_quality_flag | string | — | "observed" \| "interpolated" \| "missing" | derived from is_observed/is_interpolated | — |
| energy_kwh | float | kWh | Energy represented by the interval | analysis_demand_kw × interval_minutes / 60 | NaN propagates from analysis_demand_kw |
| interval_minutes | int | min | Native interval duration | detected or configured | — |
| date | date | — | Calendar date of interval | from timestamp | — |
| year/month/day | int | — | Calendar components | from timestamp | — |
| day_of_year | int | — | 1-366 | from timestamp | — |
| hour/minute | int | — | Time-of-day components | from timestamp | — |
| day_of_week | int | — | Monday=0 | from timestamp | — |
| day_name | string | — | e.g. "Monday" | from timestamp | — |
| is_weekday / is_weekend | bool | — | Calendar weekday status (holiday-independent) | day_of_week < 5 | — |
| season | string | — | winter/spring/summer/fall | month → calendar.seasons config, default meteorological | — |
| day_type | string | — | weekday \| weekend \| holiday | holiday overrides weekday/weekend | — |
| n_meters_reporting | int | — | Count of meters with non-null analysis_demand_kw at a timestamp, for an aggregated entity | notna().sum() across constituent meters | 0 if none reporting |

## Daily profile / feature fields

| Field | Unit | Meaning | Calculation |
|---|---|---|---|
| entity_id | — | meter_id, group name, or "Portfolio" | config-derived |
| is_complete_day | bool | True only if interval count == expected AND none NaN | see profiles.construct_daily_profiles |
| normalized_demand | — | demand_kw / daily peak demand_kw | NaN if peak is 0 or NaN |
| mean_demand_kw / maximum_demand_kw / minimum_demand_kw | kW | Daily stats | pandas mean/max/min, skipna |
| daily_energy_kwh | kWh | Sum of interval energy for the day | Σ demand_kw × interval_minutes/60 |
| peak_time | time | Time-of-day of daily maximum | idxmax of demand_kw within day |
| load_factor | — | mean/max | NaN if max ≤ 0 |
| peak_to_average_ratio | — | max/mean | NaN if mean ≤ 0 |
| standard_deviation_kw | kW | Daily std of demand | pandas std |
| coefficient_of_variation | — | std/mean | NaN if mean ≤ 0 |

## Valid values

- `data_quality_flag`: `observed`, `interpolated`, `missing`
- `day_type`: `weekday`, `weekend`, `holiday` (custom_day_type is a Stage 2+ extension point)
- `season`: as configured in `calendar.seasons`, default `winter`/`spring`/`summer`/`fall`
