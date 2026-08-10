"""
Load-shape classification.

Implements Specification Section 18: independent rule-based shape flags
plus one primary shape classification per daily profile. HEURISTIC
method (Section 48) -- rule-based, not statistical; documented
thresholds are configurable via the `thresholds` parameter.

Statistical (clustering-based) shape classification is a Stage 2
extension available via clustering.py; this module is the rule-based
baseline required as the initial implementation (Section 18: "may
initially use rule-based methods... architecture must permit later
statistical classification").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "flat_cv_max": 0.15,             # coefficient of variation below which a day is "flat"
    "highly_peaked_ratio_min": 2.0,  # peak-to-average ratio above which "highly peaked"
    "sharp_peak_width_frac_max": 0.10,  # peak duration (fraction of day) below which "sharp"
    "sustained_high_frac_min": 0.30,    # fraction of day >= 90% of daily peak -> sustained high load
    "sustained_high_pct_of_peak": 0.90,
    "overnight_heavy_ratio_min": 1.1,   # overnight_mean / daytime_mean above which overnight-heavy
    "multi_peak_min_count": 2,
}


def _segment_has_peak(daily_profile: pd.DataFrame, hour_start: int, hour_end: int) -> bool:
    seg = daily_profile[
        (daily_profile["hour"] >= hour_start) & (daily_profile["hour"] < hour_end)
    ]
    if seg.empty or seg["demand_kw"].isna().all():
        return False
    day_max = daily_profile["demand_kw"].max()
    if not day_max or day_max <= 0 or pd.isna(day_max):
        return False
    seg_max = seg["demand_kw"].max()
    return bool(seg_max >= 0.85 * day_max)  # segment contains the (near-)daily peak


def classify_daily_shape(
    daily_profiles: pd.DataFrame,
    daily_features: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Compute independent shape flags and one primary classification per
    (entity_id, date).

    Parameters
    ----------
    daily_profiles:
        Output of profiles.construct_daily_profiles (needs entity_id,
        date, time_of_day, demand_kw). is_peak/is_valley columns from
        peaks.detect_local_peaks_valleys, if present, refine
        is_multi_peak; otherwise multi-peak falls back to a demand-level
        heuristic.
    daily_features:
        Output of profiles.calculate_daily_features, providing
        maximum_demand_kw, mean_demand_kw (via peak_to_average_ratio),
        coefficient_of_variation.
    thresholds:
        Overrides for DEFAULT_THRESHOLDS.

    Returns
    -------
    pandas.DataFrame
        One row per (entity_id, date): all boolean flags
        (is_flat, is_highly_peaked, has_morning_peak, has_midday_peak,
        has_afternoon_peak, has_evening_peak, is_overnight_heavy,
        is_multi_peak, has_sharp_peak, has_sustained_high_load,
        has_peak_valley_pattern, is_unusual) plus primary_shape (string).

    Algorithm
    ---------
    Each flag is an independent rule evaluated against
    thresholds/DEFAULT_THRESHOLDS. primary_shape is assigned by a fixed
    priority order (first matching rule wins): highly_peaked with a
    single sharp peak in a named segment -> "<segment>_peak"; multi_peak
    -> "multi_peak"; overnight_heavy -> "overnight_heavy"; flat ->
    "flat"; else "mixed/other". is_unusual is set when insufficient data
    (< 50% of expected intervals present) rather than left unclassified.

    HEURISTIC method (Section 48) -- rule thresholds are configurable,
    not learned.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    dp = daily_profiles.copy()
    dp["hour"] = dp["time_of_day"].apply(lambda t: t.hour if pd.notna(t) else np.nan)

    feats = daily_features.set_index(["entity_id", "date"])

    records = []
    for (entity_id, date), g in dp.groupby(["entity_id", "date"]):
        key = (entity_id, date)
        if key not in feats.index:
            continue
        f = feats.loc[key]
        n_present = g["demand_kw"].notna().sum()
        n_total = len(g)
        insufficient_data = n_total == 0 or (n_present / n_total) < 0.5

        cv = f.get("coefficient_of_variation", np.nan)
        peak_to_avg = f.get("peak_to_average_ratio", np.nan)
        day_max = f.get("maximum_demand_kw", np.nan)

        is_flat = bool(pd.notna(cv) and cv <= th["flat_cv_max"])
        is_highly_peaked = bool(pd.notna(peak_to_avg) and peak_to_avg >= th["highly_peaked_ratio_min"])

        has_morning_peak = _segment_has_peak(g, 6, 10)
        has_midday_peak = _segment_has_peak(g, 10, 14)
        has_afternoon_peak = _segment_has_peak(g, 14, 18)
        has_evening_peak = _segment_has_peak(g, 18, 22)

        overnight = g[g["hour"].isin(set(range(22, 24)) | set(range(0, 6)))]["demand_kw"].mean()
        daytime = g[g["hour"].isin(range(6, 22))]["demand_kw"].mean()
        is_overnight_heavy = bool(
            pd.notna(overnight) and pd.notna(daytime) and daytime > 0
            and (overnight / daytime) >= th["overnight_heavy_ratio_min"]
        )

        if "is_peak" in g.columns:
            n_local_peaks = int(g["is_peak"].sum())
        else:
            n_local_peaks = int(sum([has_morning_peak, has_midday_peak, has_afternoon_peak, has_evening_peak]))
        is_multi_peak = n_local_peaks >= th["multi_peak_min_count"]

        peak_width_frac = (
            g[g["demand_kw"] >= th["sustained_high_pct_of_peak"] * day_max].shape[0] / n_total
            if n_total and pd.notna(day_max) and day_max > 0 else np.nan
        )
        has_sharp_peak = bool(pd.notna(peak_width_frac) and peak_width_frac <= th["sharp_peak_width_frac_max"])
        has_sustained_high_load = bool(
            pd.notna(peak_width_frac) and peak_width_frac >= th["sustained_high_frac_min"]
        )

        has_peak_valley_pattern = bool(
            ("is_peak" in g.columns and "is_valley" in g.columns)
            and g["is_peak"].sum() >= 1 and g["is_valley"].sum() >= 1
        )

        is_unusual = bool(insufficient_data)

        # primary classification: first matching rule wins
        if is_unusual:
            primary_shape = "insufficient_data"
        elif is_highly_peaked and has_sharp_peak and has_morning_peak and not (has_afternoon_peak or has_evening_peak):
            primary_shape = "morning_peak"
        elif is_highly_peaked and has_sharp_peak and has_afternoon_peak and not has_evening_peak:
            primary_shape = "afternoon_peak"
        elif is_highly_peaked and has_sharp_peak and has_evening_peak:
            primary_shape = "evening_peak"
        elif is_highly_peaked and has_sharp_peak and has_midday_peak:
            primary_shape = "midday_peak"
        elif is_multi_peak:
            primary_shape = "multi_peak"
        elif is_overnight_heavy:
            primary_shape = "overnight_heavy"
        elif is_flat:
            primary_shape = "flat"
        else:
            primary_shape = "mixed_other"

        records.append(
            {
                "entity_id": entity_id, "date": date,
                "is_flat": is_flat, "is_highly_peaked": is_highly_peaked,
                "has_morning_peak": has_morning_peak, "has_midday_peak": has_midday_peak,
                "has_afternoon_peak": has_afternoon_peak, "has_evening_peak": has_evening_peak,
                "is_overnight_heavy": is_overnight_heavy, "is_multi_peak": is_multi_peak,
                "has_sharp_peak": has_sharp_peak, "has_sustained_high_load": has_sustained_high_load,
                "has_peak_valley_pattern": has_peak_valley_pattern, "is_unusual": is_unusual,
                "primary_shape": primary_shape,
            }
        )
    return pd.DataFrame.from_records(records)
