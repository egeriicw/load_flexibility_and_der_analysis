"""
Daily profile construction and daily feature calculation.

Implements Specification Section 8 (Incomplete Days), Section 10 (Daily
Profile), Section 11 (Absolute and Normalized Profiles), and the Stage-1
subset of Section 12 (Daily Features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_intervals_per_day(interval_minutes: int) -> int:
    """Number of intervals in a conventional 00:00-23:59 analytical day."""
    return int(24 * 60 / interval_minutes)


def construct_daily_profiles(
    entity_df: pd.DataFrame, interval_minutes: int, entity_id: str
) -> pd.DataFrame:
    """
    Construct one row per (date, interval-of-day) analytical daily profile
    for a single entity (meter, group, or portfolio).

    Parameters
    ----------
    entity_df:
        DataFrame with columns timestamp, demand_kw (already aggregated
        to the entity level by entities.aggregate_entity_load, or a
        single meter's analysis_demand_kw renamed to demand_kw).
    interval_minutes:
        Native interval duration.
    entity_id:
        Identifier for the meter/group/portfolio this profile belongs to
        (attached as a column for traceability, Section 2.5).

    Returns
    -------
    pandas.DataFrame
        Columns: entity_id, date, interval_index (0..N-1 within day),
        time_of_day, demand_kw, is_complete_day (attached per-date, same
        value repeated across that date's rows for convenience).

    Algorithm
    ---------
    Groups by calendar date. A day is flagged complete only if it
    contains the expected number of intervals for the native resolution
    AND none of the demand_kw values were missing after Stage 1's
    missing-data step (a day with 96 present-but-still-NaN rows -- e.g.
    a gap larger than max_interpolation_intervals -- is NOT complete).

    Assumptions
    -----------
    Per Section 8, incomplete days are retained (not dropped) here;
    filtering by completeness is left to the caller/config policy
    (data.incomplete_days.policy), since this function's job is only to
    detect and flag, not to exclude.
    """
    df = entity_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["time_of_day"] = df["timestamp"].dt.time
    expected_n = expected_intervals_per_day(interval_minutes)

    df = df.sort_values(["date", "timestamp"])
    df["interval_index"] = df.groupby("date").cumcount()

    completeness = df.groupby("date").agg(
        n_rows=("demand_kw", "size"),
        n_present=("demand_kw", lambda s: s.notna().sum()),
    )
    completeness["is_complete_day"] = (completeness["n_rows"] == expected_n) & (
        completeness["n_present"] == expected_n
    )
    df = df.merge(completeness["is_complete_day"], on="date", how="left")
    df["entity_id"] = entity_id

    return df[
        ["entity_id", "date", "interval_index", "time_of_day", "demand_kw", "is_complete_day"]
    ]


def normalize_daily_profiles(daily_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Add a peak-normalized demand column to a daily profile table.

    Algorithm
    ---------
        normalized_demand = demand_kw / daily_peak_demand_kw

    where daily_peak_demand_kw is the max demand_kw for that
    (entity_id, date). This is the "peak_normalized" method documented
    in Section 11; other normalization methods (min-max, z-score, mean,
    energy) are extension points not implemented in Stage 1.

    Edge Cases
    ----------
    A day whose peak demand is 0 or NaN produces NaN normalized values
    for that day (division-by-zero avoided explicitly, not left to
    produce inf).
    """
    out = daily_profiles.copy()
    daily_peak = out.groupby(["entity_id", "date"])["demand_kw"].transform("max")
    out["normalized_demand"] = np.where(
        (daily_peak > 0) & daily_peak.notna(), out["demand_kw"] / daily_peak, np.nan
    )
    return out


def calculate_daily_features(
    daily_profiles: pd.DataFrame, interval_minutes: int
) -> pd.DataFrame:
    """
    Calculate per-day summary features (Stage 1 subset of Section 12).

    Parameters
    ----------
    daily_profiles:
        Output of construct_daily_profiles for one entity.
    interval_minutes:
        Native interval duration, used for energy and peak_time lookup.

    Returns
    -------
    pandas.DataFrame
        One row per (entity_id, date) with: mean_demand_kw,
        maximum_demand_kw, minimum_demand_kw, daily_energy_kwh,
        peak_time, load_factor, peak_to_average_ratio,
        standard_deviation_kw, coefficient_of_variation, is_complete_day.

    Formulas
    --------
    daily_energy_kwh = sum(demand_kw) * interval_minutes / 60
    load_factor = mean_demand_kw / maximum_demand_kw   (0 if peak is 0)
    peak_to_average_ratio = maximum_demand_kw / mean_demand_kw (NaN if mean is 0)
    coefficient_of_variation = std_kw / mean_demand_kw (NaN if mean is 0)

    Missing-data Behavior
    ----------------------
    All aggregations use skipna (pandas default); a day with missing
    intervals still produces features from whatever is present, but
    is_complete_day flags that the features rest on partial data.
    """
    def peak_time_for_group(g: pd.DataFrame):
        if g["demand_kw"].notna().sum() == 0:
            return pd.NaT
        idx = g["demand_kw"].idxmax()
        return g.loc[idx, "time_of_day"]

    columns = [
        "entity_id",
        "date",
        "mean_demand_kw",
        "maximum_demand_kw",
        "minimum_demand_kw",
        "daily_energy_kwh",
        "peak_time",
        "load_factor",
        "peak_to_average_ratio",
        "standard_deviation_kw",
        "coefficient_of_variation",
        "is_complete_day",
    ]
    if daily_profiles.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for (entity_id, date), g in daily_profiles.groupby(["entity_id", "date"]):
        mean_kw = g["demand_kw"].mean()
        max_kw = g["demand_kw"].max()
        min_kw = g["demand_kw"].min()
        std_kw = g["demand_kw"].std()
        energy_kwh = g["demand_kw"].sum(skipna=True) * interval_minutes / 60.0
        load_factor = (mean_kw / max_kw) if max_kw and max_kw > 0 else np.nan
        peak_to_avg = (max_kw / mean_kw) if mean_kw and mean_kw > 0 else np.nan
        cv = (std_kw / mean_kw) if mean_kw and mean_kw > 0 else np.nan
        records.append(
            {
                "entity_id": entity_id,
                "date": date,
                "mean_demand_kw": mean_kw,
                "maximum_demand_kw": max_kw,
                "minimum_demand_kw": min_kw,
                "daily_energy_kwh": energy_kwh,
                "peak_time": peak_time_for_group(g),
                "load_factor": load_factor,
                "peak_to_average_ratio": peak_to_avg,
                "standard_deviation_kw": std_kw,
                "coefficient_of_variation": cv,
                "is_complete_day": g["is_complete_day"].iloc[0],
            }
        )
    return pd.DataFrame.from_records(records)
