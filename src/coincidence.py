"""
Meter coincidence analysis.

Implements Specification Section 22: which meters contribute to
portfolio/group peaks, simultaneous peaking, offset peaks, percentage
contribution to aggregate peak, and threshold coincidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_peak_contribution(
    obs_df: pd.DataFrame, meter_ids: list[str], top_n: int = 10, demand_col: str = "analysis_demand_kw"
) -> pd.DataFrame:
    """
    For the top-N portfolio/entity peak intervals, calculate each
    meter's demand and percentage contribution.

    Parameters
    ----------
    obs_df:
        Native-resolution observations for all meters (columns:
        timestamp, meter_id, demand_col).
    meter_ids:
        Meters comprising the entity whose peaks are being examined.
    top_n:
        Number of top aggregate-demand intervals to examine.
    demand_col:
        Column holding the demand value to sum/contribute.

    Returns
    -------
    pandas.DataFrame
        One row per (timestamp, meter_id) for the top_n aggregate
        intervals: timestamp, aggregate_demand_kw, meter_id,
        meter_demand_kw, pct_contribution.

    Algorithm
    ---------
    1. Sum demand_col across meter_ids per timestamp (aggregate).
    2. Take the top_n timestamps by aggregate demand.
    3. For each such timestamp, report each meter's demand and its
       percentage of the aggregate at that instant.

    Missing-data Behavior
    ----------------------
    A meter with NaN demand at a peak timestamp contributes 0 to the
    percentage denominator's numerator but its row shows NaN
    meter_demand_kw / pct_contribution (not silently 0), so
    non-reporting is visible.
    """
    subset = obs_df[obs_df["meter_id"].isin(meter_ids)]
    agg = subset.groupby("timestamp")[demand_col].sum(min_count=1).rename("aggregate_demand_kw")
    top_timestamps = agg.sort_values(ascending=False).head(top_n).index

    records = []
    for ts in top_timestamps:
        agg_val = agg.loc[ts]
        at_ts = subset[subset["timestamp"] == ts]
        for _, row in at_ts.iterrows():
            pct = (row[demand_col] / agg_val * 100) if pd.notna(row[demand_col]) and agg_val else np.nan
            records.append(
                {
                    "timestamp": ts,
                    "aggregate_demand_kw": agg_val,
                    "meter_id": row["meter_id"],
                    "meter_demand_kw": row[demand_col],
                    "pct_contribution": pct,
                }
            )
    return pd.DataFrame.from_records(records)


def calculate_interval_coincidence(
    obs_df: pd.DataFrame, meter_ids: list[str], threshold_pct_of_peak: float = 0.9, demand_col: str = "analysis_demand_kw"
) -> pd.DataFrame:
    """
    For each meter, determine how often it is near its own peak (>=
    threshold_pct_of_peak of that meter's own max demand) AT THE SAME
    TIME another meter in the group is also near its own peak.

    Parameters
    ----------
    obs_df, meter_ids, demand_col:
        See calculate_peak_contribution.
    threshold_pct_of_peak:
        Fraction of each meter's own maximum demand considered "near
        peak" for that meter (e.g. 0.9 = within 10% of its own peak).

    Returns
    -------
    pandas.DataFrame
        meter_id, n_near_peak_intervals, n_coincident_with_any_other,
        coincidence_rate (n_coincident / n_near_peak, NaN if
        n_near_peak is 0).

    Algorithm
    ---------
    Build a boolean "near own peak" matrix (timestamp x meter). For each
    meter's near-peak intervals, coincidence_rate = fraction of those
    intervals where >=1 other configured meter is also near its own
    peak at that same timestamp.
    """
    subset = obs_df[obs_df["meter_id"].isin(meter_ids)]
    pivot = subset.pivot_table(index="timestamp", columns="meter_id", values=demand_col)
    near_peak = pivot.apply(lambda col: col >= threshold_pct_of_peak * col.max(skipna=True), axis=0)

    records = []
    for meter in meter_ids:
        if meter not in near_peak.columns:
            records.append({"meter_id": meter, "n_near_peak_intervals": 0, "n_coincident_with_any_other": 0, "coincidence_rate": np.nan})
            continue
        own_near_peak = near_peak[meter].fillna(False)
        others = near_peak.drop(columns=[meter], errors="ignore").fillna(False)
        any_other_near_peak = others.any(axis=1) if not others.empty else pd.Series(False, index=near_peak.index)
        coincident = own_near_peak & any_other_near_peak
        n_near_peak = int(own_near_peak.sum())
        n_coincident = int(coincident.sum())
        records.append(
            {
                "meter_id": meter,
                "n_near_peak_intervals": n_near_peak,
                "n_coincident_with_any_other": n_coincident,
                "coincidence_rate": round(n_coincident / n_near_peak, 3) if n_near_peak else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def calculate_diversity_factor(
    obs_df: pd.DataFrame, meter_ids: list[str], demand_col: str = "analysis_demand_kw"
) -> dict:
    """
    Calculate the diversity factor: how much aggregation reduces
    apparent peak coincidence, defined as:

        diversity_factor = sum(individual meter peak demands)
                            / (peak of the summed/aggregate demand)

    A diversity_factor > 1 means individual peaks do not all occur
    simultaneously (aggregation "smooths" the portfolio peak below the
    sum of individual peaks).

    Returns
    -------
    dict: sum_of_individual_peaks_kw, aggregate_peak_kw, diversity_factor
    """
    subset = obs_df[obs_df["meter_id"].isin(meter_ids)]
    individual_peaks = subset.groupby("meter_id")[demand_col].max()
    sum_individual_peaks = float(individual_peaks.sum())

    agg = subset.groupby("timestamp")[demand_col].sum(min_count=1)
    aggregate_peak = float(agg.max()) if len(agg) else np.nan

    diversity_factor = (
        sum_individual_peaks / aggregate_peak if aggregate_peak and aggregate_peak > 0 else np.nan
    )
    return {
        "sum_of_individual_peaks_kw": sum_individual_peaks,
        "aggregate_peak_kw": aggregate_peak,
        "diversity_factor": diversity_factor,
    }
