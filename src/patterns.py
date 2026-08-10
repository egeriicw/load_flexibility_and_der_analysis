"""
Pattern discovery.

Implements Specification Section 21: identify recurring behavior not
explicitly specified by the user. HEURISTIC/STATISTICAL method
(Section 48). Every discovered pattern reports frequency, dates,
magnitude, and statistical support -- never claims physical causation
(Section 21: "Do not describe statistical association as physical
causation").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def discover_recurring_peak_timing(
    daily_features: pd.DataFrame, entity_id: str, min_occurrences: int = 3, window_minutes: int = 30
) -> pd.DataFrame:
    """
    Identify a recurring peak_time-of-day pattern for an entity.

    Parameters
    ----------
    daily_features:
        Output of profiles.calculate_daily_features (needs entity_id,
        date, peak_time, is_complete_day).
    entity_id:
        Entity to analyze.
    min_occurrences:
        Minimum number of days a peak-time cluster must appear on to be
        reported as a pattern.
    window_minutes:
        Tolerance window (minutes) for grouping similar peak_time values
        into the same recurring-timing bucket.

    Returns
    -------
    pandas.DataFrame
        pattern_id, pattern_type="recurring_peak_timing", description,
        frequency, dates (list), representative_peak_time,
        statistical_support (fraction of complete days matching).

    Algorithm
    ---------
    Convert peak_time to minutes-since-midnight, round to the nearest
    window_minutes bucket, count occurrences per bucket, report buckets
    meeting min_occurrences.
    """
    feats = daily_features[
        (daily_features["entity_id"] == entity_id) & (daily_features["is_complete_day"])
    ].dropna(subset=["peak_time"])
    if feats.empty:
        return pd.DataFrame()

    minutes = feats["peak_time"].apply(lambda t: t.hour * 60 + t.minute)
    bucket = (minutes // window_minutes) * window_minutes
    feats = feats.assign(_bucket=bucket)

    records = []
    n_complete = len(feats)
    for i, (bucket_val, g) in enumerate(feats.groupby("_bucket")):
        if len(g) < min_occurrences:
            continue
        hh, mm = divmod(int(bucket_val), 60)
        records.append(
            {
                "pattern_id": f"{entity_id}_peak_timing_{i:03d}",
                "pattern_type": "recurring_peak_timing",
                "description": f"Daily peak recurs near {hh:02d}:{mm:02d} on {len(g)} of {n_complete} complete days",
                "frequency": len(g),
                "dates": sorted(g["date"].tolist()),
                "representative_peak_time": f"{hh:02d}:{mm:02d}",
                "statistical_support": round(len(g) / n_complete, 3) if n_complete else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def discover_recurring_shapes(shape_classifications: pd.DataFrame, entity_id: str, min_occurrences: int = 3) -> pd.DataFrame:
    """
    Identify recurring primary_shape classifications for an entity.

    Parameters
    ----------
    shape_classifications:
        Output of shapes.classify_daily_shape.
    entity_id:
        Entity to analyze.
    min_occurrences:
        Minimum days a shape must occur on to be reported.

    Returns
    -------
    pandas.DataFrame: pattern_id, pattern_type="recurring_shape",
    description, frequency, dates, shape, statistical_support.
    """
    sub = shape_classifications[shape_classifications["entity_id"] == entity_id]
    if sub.empty:
        return pd.DataFrame()
    n_total = len(sub)

    records = []
    for i, (shape_val, g) in enumerate(sub.groupby("primary_shape")):
        if len(g) < min_occurrences or shape_val in ("insufficient_data",):
            continue
        records.append(
            {
                "pattern_id": f"{entity_id}_shape_{i:03d}",
                "pattern_type": "recurring_shape",
                "description": f"'{shape_val}' shape recurs on {len(g)} of {n_total} days",
                "frequency": len(g),
                "dates": sorted(g["date"].tolist()),
                "shape": shape_val,
                "statistical_support": round(len(g) / n_total, 3) if n_total else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def discover_outlier_days(
    daily_features: pd.DataFrame, entity_id: str, z_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Identify unusual/outlier days by z-score of daily_energy_kwh and
    maximum_demand_kw against the entity's own complete-day distribution.

    Parameters
    ----------
    daily_features:
        Output of profiles.calculate_daily_features.
    entity_id:
        Entity to analyze.
    z_threshold:
        Absolute z-score above which a day is flagged as an outlier.

    Returns
    -------
    pandas.DataFrame: pattern_id, pattern_type="outlier_day",
    description, date, metric, value, z_score, statistical_support
    ("z_score_threshold").

    Algorithm
    ---------
    STATISTICAL method: z = (x - mean) / std computed separately for
    daily_energy_kwh and maximum_demand_kw over complete days only.
    Requires >= 5 complete days to compute a meaningful std; fewer
    returns an empty result.
    """
    feats = daily_features[
        (daily_features["entity_id"] == entity_id) & (daily_features["is_complete_day"])
    ]
    if len(feats) < 5:
        return pd.DataFrame()

    records = []
    idx = 0
    for metric in ("daily_energy_kwh", "maximum_demand_kw"):
        mean, std = feats[metric].mean(), feats[metric].std()
        if not std or pd.isna(std) or std == 0:
            continue
        z = (feats[metric] - mean) / std
        outliers = feats[z.abs() >= z_threshold]
        for _, row in outliers.iterrows():
            records.append(
                {
                    "pattern_id": f"{entity_id}_outlier_{idx:03d}",
                    "pattern_type": "outlier_day",
                    "description": f"{row['date']} is an outlier on {metric} (z={z.loc[row.name]:.2f})",
                    "date": row["date"],
                    "metric": metric,
                    "value": row[metric],
                    "z_score": round(float(z.loc[row.name]), 3),
                    "statistical_support": f"z >= {z_threshold}",
                }
            )
            idx += 1
    return pd.DataFrame.from_records(records)
