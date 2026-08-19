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


def _r_squared(d: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((d - pred) ** 2)
    ss_tot = np.sum((d - d.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def fit_change_point_model_2p(temperature_f: np.ndarray, demand_kw: np.ndarray, min_points: int = 4) -> dict:
    """
    Fit the 2-parameter (no-breakpoint) plain-linear model:

        demand(T) = base + slope*T

    Purpose
    -------
    Section 13: the no-breakpoint member of the ASHRAE GL14 / IPMVP
    change-point model family. Serves as the simplest baseline in
    select_best_change_point_model -- if a meter's temperature response
    is genuinely linear (or absent), the added breakpoint parameters of
    the 3P/4P/5P models should not be preferred just because they can
    only fit at least as well.

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays. NaN pairs are dropped before fitting.
    min_points:
        Minimum valid paired observations required (needs only enough
        degrees of freedom for a 2-parameter OLS fit).

    Returns
    -------
    dict with keys: base_kw, slope_kw_per_f, r_squared, n_points,
    method="change_point_regression_2p", success (bool).
    """
    t = np.asarray(temperature_f, dtype=float)
    d = np.asarray(demand_kw, dtype=float)
    mask = np.isfinite(t) & np.isfinite(d)
    t, d = t[mask], d[mask]

    if len(t) < min_points:
        return {
            "base_kw": np.nan, "slope_kw_per_f": np.nan,
            "r_squared": np.nan, "n_points": len(t),
            "method": "change_point_regression_2p", "success": False,
        }

    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred = X @ coef

    return {
        "base_kw": float(coef[0]),
        "slope_kw_per_f": float(coef[1]),
        "r_squared": _r_squared(d, pred),
        "n_points": len(t),
        "method": "change_point_regression_2p",
        "success": True,
    }


def fit_change_point_model_3p_heating(
    temperature_f: np.ndarray, demand_kw: np.ndarray, min_points: int = 5
) -> dict:
    """
    Fit a 3-parameter heating change-point (balance-point) model:

        demand(T) = baseload + slope*(breakpoint-T)  if T < breakpoint
        demand(T) = baseload                          if T >= breakpoint

    Purpose
    -------
    Section 13: the heating-side counterpart to fit_change_point_model's
    cooling-only 3P model, for entities whose demand rises as
    temperature FALLS below a balance point (electric-resistance/heat-
    pump heating, or a fossil-fuel meter) rather than as it rises above
    one.

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays (typically daily mean temperature vs. daily mean
        demand). NaN pairs are dropped before fitting.
    min_points:
        Minimum valid paired observations required to attempt a fit.

    Returns
    -------
    dict with keys: baseload_kw, slope_kw_per_f, breakpoint_f, r_squared,
    n_points, method="change_point_regression_3p_heating", success
    (bool). If fewer than min_points valid points remain, returns
    success=False and NaN parameters rather than fitting a meaningless
    model.

    Algorithm
    ---------
    Grid-search the breakpoint over the observed temperature range (1
    degree steps); for each candidate, fit baseload+slope by OLS on the
    piecewise-linear design and keep the breakpoint with lowest SSE.
    Mirrors fit_change_point_model exactly, with the excess term flipped
    to the heating side.

    Assumptions
    -----------
    This is a STATISTICAL/MODELED relationship, not a causal claim
    (Section 13). Same weekday-only-input convention as
    fit_change_point_model applies.
    """
    t = np.asarray(temperature_f, dtype=float)
    d = np.asarray(demand_kw, dtype=float)
    mask = np.isfinite(t) & np.isfinite(d)
    t, d = t[mask], d[mask]

    empty = {
        "baseload_kw": np.nan, "slope_kw_per_f": np.nan, "breakpoint_f": np.nan,
        "r_squared": np.nan, "n_points": len(t),
        "method": "change_point_regression_3p_heating", "success": False,
    }
    if len(t) < min_points:
        return empty

    candidates = np.arange(np.floor(t.min()) + 1, np.ceil(t.max()), 1.0)
    if len(candidates) < 1:
        return empty

    best = None
    for bp in candidates:
        excess = np.maximum(bp - t, 0)
        X = np.column_stack([np.ones_like(excess), excess])
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        pred = X @ coef
        sse = float(np.sum((d - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"sse": sse, "breakpoint_f": float(bp), "baseload_kw": float(coef[0]), "slope_kw_per_f": float(coef[1])}

    excess = np.maximum(best["breakpoint_f"] - t, 0)
    pred = best["baseload_kw"] + best["slope_kw_per_f"] * excess

    return {
        "baseload_kw": best["baseload_kw"],
        "slope_kw_per_f": best["slope_kw_per_f"],
        "breakpoint_f": best["breakpoint_f"],
        "r_squared": _r_squared(d, pred),
        "n_points": len(t),
        "method": "change_point_regression_3p_heating",
        "success": True,
    }


# Alias for naming symmetry with fit_change_point_model_3p_heating; both
# 3P models in the family are referenced by this name elsewhere (e.g.
# select_best_change_point_model).
fit_change_point_model_3p_cooling = fit_change_point_model


def fit_change_point_model_4p(
    temperature_f: np.ndarray, demand_kw: np.ndarray, min_points: int = 8
) -> dict:
    """
    Fit a 4-parameter heating+cooling change-point model sharing a
    SINGLE breakpoint:

        demand(T) = base + hsl*max(cp-T, 0) + csl*max(T-cp, 0)

    Purpose
    -------
    Section 13 extension point: the middle ground between the 3P
    cooling/heating models (one slope) and the 5P model (two
    independent breakpoints) -- useful when heating and cooling both
    show up but the data doesn't support resolving two distinct
    breakpoints (Section 13's 5P model needs more points; the 4P model
    needs only one breakpoint to be grid-searched).

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays. NaN pairs are dropped before fitting.
    min_points:
        Minimum valid paired observations required to attempt a fit.

    Returns
    -------
    dict with keys: base_kw, heating_slope_kw_per_f, cooling_slope_kw_per_f,
    breakpoint_f, r_squared, n_points, method="change_point_regression_4p",
    success (bool).

    Algorithm
    ---------
    Grid-search the single breakpoint over the observed temperature
    range (1 degree steps); for each candidate, fit
    base/heating-slope/cooling-slope by bounded least squares (slopes
    constrained >= 0) and keep the breakpoint with lowest SSE.

    Assumptions
    -----------
    This is a STATISTICAL/MODELED relationship, not a causal claim
    (Section 13). Same weekday-only-input convention as
    fit_change_point_model applies.
    """
    t = np.asarray(temperature_f, dtype=float)
    d = np.asarray(demand_kw, dtype=float)
    mask = np.isfinite(t) & np.isfinite(d)
    t, d = t[mask], d[mask]

    empty = {
        "base_kw": np.nan, "heating_slope_kw_per_f": np.nan,
        "cooling_slope_kw_per_f": np.nan, "breakpoint_f": np.nan,
        "r_squared": np.nan, "n_points": len(t),
        "method": "change_point_regression_4p", "success": False,
    }
    if len(t) < min_points:
        return empty

    candidates = np.arange(np.floor(t.min()) + 1, np.ceil(t.max()), 1.0)
    if len(candidates) < 1:
        return empty

    best = None
    for cp in candidates:
        heating_excess = np.maximum(cp - t, 0)
        cooling_excess = np.maximum(t - cp, 0)
        if not heating_excess.any() and not cooling_excess.any():
            continue  # degenerate: pure baseload, no slope information
        X = np.column_stack([np.ones_like(t), heating_excess, cooling_excess])
        fit = optimize.lsq_linear(X, d, bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))
        pred = X @ fit.x
        sse = float(np.sum((d - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "sse": sse, "cp": float(cp),
                "base": float(fit.x[0]), "hsl": float(fit.x[1]), "csl": float(fit.x[2]),
            }

    if best is None:
        return empty

    heating_excess = np.maximum(best["cp"] - t, 0)
    cooling_excess = np.maximum(t - best["cp"], 0)
    pred = best["base"] + best["hsl"] * heating_excess + best["csl"] * cooling_excess

    return {
        "base_kw": best["base"],
        "heating_slope_kw_per_f": best["hsl"],
        "cooling_slope_kw_per_f": best["csl"],
        "breakpoint_f": best["cp"],
        "r_squared": _r_squared(d, pred),
        "n_points": len(t),
        "method": "change_point_regression_4p",
        "success": True,
    }


def fit_change_point_model_5p(
    temperature_f: np.ndarray, demand_kw: np.ndarray, min_points: int = 10
) -> dict:
    """
    Fit a 5-parameter heating+cooling change-point model:

        demand(T) = base + hsl*max(hcp-T, 0) + csl*max(T-ccp, 0),  hcp <= ccp

    Purpose
    -------
    Section 13 extension point: the heating+cooling generalization of
    fit_change_point_model's 3-parameter cooling-only baseline, following
    the same ASHRAE GL14 / IPMVP-style change-point regression family
    (this is the 5P model; the 3P model is the special case hsl=0).
    Useful for entities with a visible heating response (electric
    resistance/heat-pump heating, or a fossil-fuel meter) in addition to
    cooling.

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays (typically daily mean temperature vs. daily mean
        demand). NaN pairs are dropped before fitting.
    min_points:
        Minimum valid paired observations required to attempt a fit.
        Higher than the 3P model's threshold because there are two
        breakpoints to resolve rather than one.

    Returns
    -------
    dict with keys: base_kw, heating_slope_kw_per_f, heating_breakpoint_f,
    cooling_slope_kw_per_f, cooling_breakpoint_f, r_squared, n_points,
    method="change_point_regression_5p", success (bool). If fewer than
    min_points valid points remain, or no breakpoint pair yields a
    non-degenerate design, returns success=False and NaN parameters
    rather than fitting a meaningless model.

    Algorithm
    ---------
    Grid-search both breakpoints jointly over the observed temperature
    range (1 degree steps, heating breakpoint <= cooling breakpoint); for
    each candidate pair, fit base/heating-slope/cooling-slope by
    bounded least squares (slopes constrained >= 0, since heating load
    must not decrease as temperature drops further below the heating
    breakpoint, and likewise for cooling) and keep the pair with lowest
    SSE.

    Assumptions
    -----------
    This is a STATISTICAL/MODELED relationship, not a causal claim
    (Section 13: "Do not imply causality simply because temperature and
    demand correlate"). Same weekday-only-input convention as
    fit_change_point_model applies -- callers should exclude weekend
    days when the weekday/weekend load-level swing would otherwise
    confound the temperature/demand relationship.
    """
    t = np.asarray(temperature_f, dtype=float)
    d = np.asarray(demand_kw, dtype=float)
    mask = np.isfinite(t) & np.isfinite(d)
    t, d = t[mask], d[mask]

    empty = {
        "base_kw": np.nan, "heating_slope_kw_per_f": np.nan, "heating_breakpoint_f": np.nan,
        "cooling_slope_kw_per_f": np.nan, "cooling_breakpoint_f": np.nan,
        "r_squared": np.nan, "n_points": len(t),
        "method": "change_point_regression_5p", "success": False,
    }
    if len(t) < min_points:
        return empty

    candidates = np.arange(np.floor(t.min()) + 1, np.ceil(t.max()), 1.0)
    if len(candidates) < 2:
        return empty

    best = None
    for hcp in candidates:
        heating_excess = np.maximum(hcp - t, 0)
        for ccp in candidates:
            if ccp < hcp:
                continue
            cooling_excess = np.maximum(t - ccp, 0)
            if not heating_excess.any() and not cooling_excess.any():
                continue  # degenerate: pure baseload, no slope information
            X = np.column_stack([np.ones_like(t), heating_excess, cooling_excess])
            fit = optimize.lsq_linear(X, d, bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))
            pred = X @ fit.x
            sse = float(np.sum((d - pred) ** 2))
            if best is None or sse < best["sse"]:
                best = {
                    "sse": sse, "hcp": float(hcp), "ccp": float(ccp),
                    "base": float(fit.x[0]), "hsl": float(fit.x[1]), "csl": float(fit.x[2]),
                }

    if best is None:
        return empty

    heating_excess = np.maximum(best["hcp"] - t, 0)
    cooling_excess = np.maximum(t - best["ccp"], 0)
    pred = best["base"] + best["hsl"] * heating_excess + best["csl"] * cooling_excess
    ss_res = np.sum((d - pred) ** 2)
    ss_tot = np.sum((d - d.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "base_kw": best["base"],
        "heating_slope_kw_per_f": best["hsl"],
        "heating_breakpoint_f": best["hcp"],
        "cooling_slope_kw_per_f": best["csl"],
        "cooling_breakpoint_f": best["ccp"],
        "r_squared": float(r_squared),
        "n_points": len(t),
        "method": "change_point_regression_5p",
        "success": True,
    }


# Parameter count per change-point model family member, used by
# select_best_change_point_model to penalize added complexity via
# adjusted R^2 rather than raw R^2 (which would trivially favor 5P).
CHANGE_POINT_MODEL_PARAM_COUNTS = {
    "2p": 2,
    "3p_heating": 3,
    "3p_cooling": 3,
    "4p": 4,
    "5p": 5,
}


def select_best_change_point_model(temperature_f: np.ndarray, demand_kw: np.ndarray) -> dict:
    """
    Fit the full ASHRAE GL14 / IPMVP change-point model family --
    2P (no breakpoint), 3P heating, 3P cooling, 4P (shared breakpoint),
    5P (independent heating+cooling breakpoints) -- and select the
    best-fitting one.

    Purpose
    -------
    Section 13: rather than assuming a priori which model form applies
    to a given meter (heating response? cooling? both? neither?), fit
    every applicable form and let the data decide which is warranted.
    This is standard LBNL/ASHRAE inverse-modeling practice; the
    individual fit_change_point_model* functions are its building
    blocks.

    Parameters
    ----------
    temperature_f, demand_kw:
        Paired arrays, as in the individual fit_* functions.

    Returns
    -------
    dict with keys:
      selected_model: one of "2p", "3p_heating", "3p_cooling", "4p",
        "5p", or None if no candidate could be fit (all failed their
        own insufficient-data guard).
      selected: the winning candidate's result dict, or None.
      candidates: dict of every candidate's own result dict (including
        ones not selected), keyed the same way, for inspection/audit.

    Algorithm
    ---------
    Each candidate is fit independently by its own function (each with
    its own insufficient-data guard and, for the multi-slope models,
    its own physical sign constraints). Selection uses ADJUSTED R^2 --
    r2_adj = 1 - (1-r2)*(n-1)/(n-p-1), where p is the candidate's
    parameter count -- rather than raw R^2, which would trivially favor
    the highest-parameter model regardless of whether the added
    breakpoints are actually warranted. The candidate with the highest
    adjusted R^2 among those that fit successfully is selected; on an
    exact tie, the simpler (fewer-parameter) model wins.

    Assumptions
    -----------
    A candidate that failed its own insufficient-data guard, or for
    which n-p-1 <= 0 (too few points to penalize), is excluded from
    selection rather than being scored as worse -- callers should treat
    an all-excluded result (selected_model=None) as "no change-point
    model could be fit," not as evidence of a flat/temperature-
    independent load. This is a STATISTICAL/MODELED comparison, not a
    causal claim (Section 13).
    """
    candidates = {
        "2p": fit_change_point_model_2p(temperature_f, demand_kw),
        "3p_heating": fit_change_point_model_3p_heating(temperature_f, demand_kw),
        "3p_cooling": fit_change_point_model_3p_cooling(temperature_f, demand_kw),
        "4p": fit_change_point_model_4p(temperature_f, demand_kw),
        "5p": fit_change_point_model_5p(temperature_f, demand_kw),
    }

    best_name = None
    best_adj_r2 = None
    best_n_params = None
    for name, result in candidates.items():
        if not result["success"]:
            continue
        r2 = result["r_squared"]
        n = result["n_points"]
        p = CHANGE_POINT_MODEL_PARAM_COUNTS[name]
        if not np.isfinite(r2) or (n - p - 1) <= 0:
            continue
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        if (
            best_adj_r2 is None
            or adj_r2 > best_adj_r2
            or (adj_r2 == best_adj_r2 and p < best_n_params)
        ):
            best_name, best_adj_r2, best_n_params = name, adj_r2, p

    return {
        "selected_model": best_name,
        "selected": candidates[best_name] if best_name is not None else None,
        "candidates": candidates,
    }
