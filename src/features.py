"""
Extended daily features and temperature analysis.

Implements Specification Section 12 (Daily Features - time-of-day
segments) and Section 13 (Temperature Analysis: bands, continuous,
change-point/balance-point regression).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


# Time-of-day segment boundaries (hour, inclusive start / exclusive end)
SEGMENTS = {
    "morning": (6, 10),
    "midday": (10, 14),
    "afternoon": (14, 18),
    "evening": (18, 22),
}
NIGHTTIME_HOURS = set(range(22, 24)) | set(range(0, 6))
DAYTIME_HOURS = set(range(6, 22))


def calculate_segment_features(daily_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-of-day segment means/peaks per (entity_id, date).

    Purpose
    -------
    Section 12: morning_peak_kw, midday_peak_kw, afternoon_peak_kw,
    evening_peak_kw, overnight_mean_kw, daytime_mean_kw, nighttime_mean_kw.

    Parameters
    ----------
    daily_profiles:
        Output of profiles.construct_daily_profiles, must contain
        entity_id, date, time_of_day, demand_kw.

    Returns
    -------
    pandas.DataFrame
        One row per (entity_id, date) with segment features.

    Algorithm
    ---------
    hour is extracted from time_of_day. Each named segment's peak is the
    max demand_kw within its [start, end) hour window; overnight/daytime/
    nighttime are means over their respective hour sets.

    Missing-data Behavior
    ----------------------
    NaN intervals are skipped (pandas skipna default); a segment with
    zero present observations returns NaN for that segment.
    """
    df = daily_profiles.copy()
    df["hour"] = df["time_of_day"].apply(lambda t: t.hour if pd.notna(t) else np.nan)

    records = []
    for (entity_id, date), g in df.groupby(["entity_id", "date"]):
        row = {"entity_id": entity_id, "date": date}
        for name, (start, end) in SEGMENTS.items():
            seg = g[(g["hour"] >= start) & (g["hour"] < end)]
            row[f"{name}_peak_kw"] = seg["demand_kw"].max() if len(seg) else np.nan
        overnight = g[g["hour"].isin(NIGHTTIME_HOURS)]
        daytime = g[g["hour"].isin(DAYTIME_HOURS)]
        row["overnight_mean_kw"] = overnight["demand_kw"].mean() if len(overnight) else np.nan
        row["nighttime_mean_kw"] = row["overnight_mean_kw"]
        row["daytime_mean_kw"] = daytime["demand_kw"].mean() if len(daytime) else np.nan
        records.append(row)
    return pd.DataFrame.from_records(records)


def calculate_temperature_daily_stats(
    obs_df: pd.DataFrame, entity_id: str
) -> pd.DataFrame:
    """
    Calculate per-day temperature statistics for an entity.

    Parameters
    ----------
    obs_df:
        Native-resolution observations for ONE meter with columns date,
        temperature_f (entity-level aggregates share site temperature
        with their constituent meter(s); pass the representative
        meter's temperature series here since temperature is site-level
        per Section 2.4, not entity-summed).
    entity_id:
        Label attached to output rows for traceability.

    Returns
    -------
    pandas.DataFrame
        entity_id, date, mean_temperature_f, maximum_temperature_f,
        minimum_temperature_f, temperature_range_f.

    Assumptions
    -----------
    If temperature_f is entirely absent, returns an empty DataFrame;
    callers must treat temperature-dependent analysis as unavailable
    (Section 2.4: "Load analysis must not fail solely because
    temperature is unavailable").
    """
    if "temperature_f" not in obs_df.columns or obs_df["temperature_f"].notna().sum() == 0:
        return pd.DataFrame(
            columns=[
                "entity_id", "date", "mean_temperature_f",
                "maximum_temperature_f", "minimum_temperature_f", "temperature_range_f",
            ]
        )
    df = obs_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    grouped = df.groupby("date")["temperature_f"].agg(["mean", "max", "min"])
    grouped["temperature_range_f"] = grouped["max"] - grouped["min"]
    grouped = grouped.rename(
        columns={"mean": "mean_temperature_f", "max": "maximum_temperature_f", "min": "minimum_temperature_f"}
    )
    grouped["entity_id"] = entity_id
    return grouped.reset_index().rename(columns={"index": "date"})[
        ["entity_id", "date", "mean_temperature_f", "maximum_temperature_f",
         "minimum_temperature_f", "temperature_range_f"]
    ]


def classify_temperature_bands(temperature_f: pd.Series, bands: list[float]) -> pd.Series:
    """
    Bin temperature into configured bands (Section 13).

    Parameters
    ----------
    temperature_f:
        Temperature values.
    bands:
        Sorted boundary list, e.g. [32, 50, 65, 80, 90] producing bins
        (-inf,32], (32,50], (50,65], (65,80], (80,90], (90,inf).

    Returns
    -------
    pandas.Series of string band labels, e.g. "32-50". NaN temperature
    yields NaN band (temperature unavailable, not an error).
    """
    edges = [-np.inf] + list(bands) + [np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_label = "below" if lo == -np.inf else str(int(lo))
        hi_label = "above" if hi == np.inf else str(int(hi))
        labels.append(f"{lo_label}-{hi_label}")
    return pd.cut(temperature_f, bins=edges, labels=labels)


def fit_change_point_model(temperature_f: np.ndarray, demand_kw: np.ndarray) -> dict:
    """
    Fit a 3-parameter cooling change-point (balance-point) model:

        demand(T) = baseload                     if T <= breakpoint
        demand(T) = baseload + slope*(T-breakpoint) if T > breakpoint

    Purpose
    -------
    Section 13: piecewise/change-point regression consistent with common
    building energy modeling practice (ASHRAE GL14 / IPMVP-style
    3-parameter cooling model). This is the DETERMINISTIC/STATISTICAL
    baseline method; more sophisticated (heating+cooling 5-parameter,
    etc.) models are extension points.

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays (typically daily mean temperature vs. daily mean
        demand). NaN pairs are dropped before fitting.

    Returns
    -------
    dict with keys: baseload_kw, slope_kw_per_f, breakpoint_f, r_squared,
    n_points, method="change_point_regression", success (bool).
    If fewer than 5 valid points remain, returns success=False and NaN
    parameters rather than fitting a meaningless model.

    Algorithm
    ---------
    Grid-search breakpoint over the observed temperature range (1 degree
    steps); for each candidate breakpoint, fit baseload+slope by OLS on
    the piecewise-linear design and keep the breakpoint with lowest SSE.

    Assumptions
    -----------
    This is a STATISTICAL/MODELED relationship, not a causal claim
    (Section 13: "Do not imply causality simply because temperature and
    demand correlate").
    """
    t = np.asarray(temperature_f, dtype=float)
    d = np.asarray(demand_kw, dtype=float)
    mask = np.isfinite(t) & np.isfinite(d)
    t, d = t[mask], d[mask]

    if len(t) < 5:
        return {
            "baseload_kw": np.nan, "slope_kw_per_f": np.nan, "breakpoint_f": np.nan,
            "r_squared": np.nan, "n_points": len(t),
            "method": "change_point_regression", "success": False,
        }

    candidates = np.arange(np.floor(t.min()) + 1, np.ceil(t.max()), 1.0)
    best = None
    for bp in candidates:
        excess = np.maximum(t - bp, 0)
        X = np.column_stack([np.ones_like(excess), excess])
        coef, residuals, *_ = np.linalg.lstsq(X, d, rcond=None)
        pred = X @ coef
        sse = float(np.sum((d - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"sse": sse, "breakpoint_f": float(bp), "baseload_kw": float(coef[0]), "slope_kw_per_f": float(coef[1])}

    excess = np.maximum(t - best["breakpoint_f"], 0)
    pred = best["baseload_kw"] + best["slope_kw_per_f"] * excess
    ss_res = np.sum((d - pred) ** 2)
    ss_tot = np.sum((d - d.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "baseload_kw": best["baseload_kw"],
        "slope_kw_per_f": best["slope_kw_per_f"],
        "breakpoint_f": best["breakpoint_f"],
        "r_squared": float(r_squared),
        "n_points": len(t),
        "method": "change_point_regression",
        "success": True,
    }
