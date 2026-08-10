"""
Demand classification, ramp detection, peak/valley detection, and peak
events.

Implements Specification Section 14 (Demand Classification), Section 15
(Peak Definitions), Section 16 (Peak Events), Section 17 (Ramps, Peaks,
Valleys). Operates on native-resolution observations for a single
entity (meter/group/portfolio), sorted by timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- Section 14
def classify_by_threshold(demand_kw: pd.Series, thresholds_kw: list[float]) -> pd.DataFrame:
    """
    Flag intervals meeting configured absolute thresholds.

    Returns
    -------
    pandas.DataFrame with one boolean column per threshold, named
    "meets_threshold_<kw>". Distinct from rank/percentile/peak
    classifications per Section 14 ("Do not collapse them into one
    generic peak flag").
    """
    out = pd.DataFrame(index=demand_kw.index)
    for th in thresholds_kw:
        out[f"meets_threshold_{int(th)}"] = demand_kw >= th
    return out


def classify_by_percentile(demand_kw: pd.Series, top_percentiles: list[float]) -> pd.DataFrame:
    """
    Flag intervals in the top-percentile of demand (e.g. top 1%, top 5%).

    Parameters
    ----------
    top_percentiles:
        Fractions, e.g. [0.99, 0.95] meaning >= the 99th / 95th
        percentile of demand_kw.

    Returns
    -------
    pandas.DataFrame with one boolean column per percentile, named
    "top_pct_<pp>" (e.g. top_pct_99).
    """
    out = pd.DataFrame(index=demand_kw.index)
    for p in top_percentiles:
        cutoff = demand_kw.quantile(p)
        out[f"top_pct_{int(round(p * 100))}"] = demand_kw >= cutoff
    return out


def classify_by_rank(demand_kw: pd.Series, top_n_hours: list[int]) -> pd.DataFrame:
    """
    Flag the top-N ranked intervals by demand (Section 14 "rank").

    Note: named top_n_hours per spec terminology; operates on whatever
    native interval the series represents (not necessarily hours).

    Returns
    -------
    pandas.DataFrame with one boolean column per N, named "top_rank_<n>".
    """
    out = pd.DataFrame(index=demand_kw.index)
    ranks = demand_kw.rank(method="first", ascending=False)
    for n in top_n_hours:
        out[f"top_rank_{n}"] = ranks <= n
    return out


# ---------------------------------------------------------------- Section 17
def detect_ramps(demand_kw: pd.Series) -> pd.DataFrame:
    """
    Calculate native-resolution interval-to-interval demand change.

    Algorithm
    ---------
        ramp_kw(t) = demand_kw(t) - demand_kw(t-1)
        ramp_up_kw(t)   = max(ramp_kw(t), 0)
        ramp_down_kw(t) = max(-ramp_kw(t), 0)

    Returns
    -------
    pandas.DataFrame: ramp_kw, ramp_up_kw, ramp_down_kw. First interval
    has NaN ramp (no prior interval). NaN input propagates NaN ramp for
    both the interval and the one following it (a ramp across a missing
    reading is not a real observed ramp).
    """
    ramp = demand_kw.diff()
    return pd.DataFrame(
        {
            "ramp_kw": ramp,
            "ramp_up_kw": ramp.clip(lower=0),
            "ramp_down_kw": (-ramp).clip(lower=0),
        },
        index=demand_kw.index,
    )


def detect_local_peaks_valleys(demand_kw: pd.Series) -> pd.DataFrame:
    """
    Identify local peaks and valleys (simple 3-point comparison).

    Algorithm
    ---------
    interval i is a local peak if demand[i] > demand[i-1] and
    demand[i] > demand[i+1]; local valley if the reverse. Boundary
    points (first/last) cannot be classified and are False.

    Returns
    -------
    pandas.DataFrame: is_peak, is_valley (bool). NaN neighbors make a
    point non-classifiable (False), since a real local extremum cannot
    be confirmed against a missing neighbor.
    """
    d = demand_kw.values
    n = len(d)
    is_peak = np.zeros(n, dtype=bool)
    is_valley = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if np.isnan(d[i - 1]) or np.isnan(d[i]) or np.isnan(d[i + 1]):
            continue
        if d[i] > d[i - 1] and d[i] > d[i + 1]:
            is_peak[i] = True
        elif d[i] < d[i - 1] and d[i] < d[i + 1]:
            is_valley[i] = True
    return pd.DataFrame({"is_peak": is_peak, "is_valley": is_valley}, index=demand_kw.index)


# ---------------------------------------------------------------- Section 16
def build_peak_events(
    obs_df: pd.DataFrame,
    meets_criterion: pd.Series,
    allowable_gap_intervals: int,
    entity_id: str,
    peak_definition: str,
) -> pd.DataFrame:
    """
    Group contiguous (within an allowable gap) qualifying intervals into
    discrete peak events.

    Parameters
    ----------
    obs_df:
        Native-resolution observations, sorted by timestamp, columns:
        timestamp, demand_kw (analysis value), temperature_f (optional),
        tou_period (optional).
    meets_criterion:
        Boolean Series aligned with obs_df.index flagging intervals that
        satisfy the peak criterion (e.g. from classify_by_threshold).
    allowable_gap_intervals:
        Maximum number of consecutive non-qualifying intervals allowed
        between two qualifying intervals for them to belong to the same
        event (Section 16).
    entity_id, peak_definition:
        Labels attached to every event for traceability (Section 15:
        "Retain separate flags" across different peak definitions).

    Returns
    -------
    pandas.DataFrame
        One row per event: event_id, entity_id, peak_definition,
        start_time, end_time, duration_hours, maximum_demand_kw,
        mean_demand_kw, minimum_demand_kw, energy_kwh (if interval
        spacing inferable), n_intervals.

    Algorithm
    ---------
    Walk qualifying interval indices in order; start a new event when
    the gap (in interval count) since the last qualifying interval
    exceeds allowable_gap_intervals; otherwise extend the current event
    to include everything between (so intra-event non-qualifying
    intervals are included in the event's own stats, consistent with it
    being a single contiguous event).
    """
    df = obs_df.reset_index(drop=True)
    qualifying_idx = np.where(meets_criterion.reset_index(drop=True).values)[0]

    events = []
    if len(qualifying_idx) == 0:
        return pd.DataFrame(
            columns=[
                "event_id", "entity_id", "peak_definition", "start_time", "end_time",
                "duration_hours", "maximum_demand_kw", "mean_demand_kw", "minimum_demand_kw",
                "n_intervals",
            ]
        )

    groups = [[qualifying_idx[0]]]
    for idx in qualifying_idx[1:]:
        gap = idx - groups[-1][-1] - 1
        if gap <= allowable_gap_intervals:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    for i, grp in enumerate(groups):
        start_i, end_i = grp[0], grp[-1]
        event_slice = df.iloc[start_i : end_i + 1]
        start_time = event_slice["timestamp"].iloc[0]
        end_time = event_slice["timestamp"].iloc[-1]
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        events.append(
            {
                "event_id": f"{entity_id}_{peak_definition}_{i:04d}",
                "entity_id": entity_id,
                "peak_definition": peak_definition,
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration_hours,
                "maximum_demand_kw": event_slice["demand_kw"].max(),
                "mean_demand_kw": event_slice["demand_kw"].mean(),
                "minimum_demand_kw": event_slice["demand_kw"].min(),
                "n_intervals": len(event_slice),
            }
        )
    return pd.DataFrame.from_records(events)


# ---------------------------------------------------------------- Section 17 (sustained)
def classify_sustained_vs_short(
    events: pd.DataFrame, sustained_threshold_hours: float
) -> pd.DataFrame:
    """
    Tag peak events as sustained vs short duration.

    Parameters
    ----------
    events:
        Output of build_peak_events (must have duration_hours).
    sustained_threshold_hours:
        Events with duration_hours >= this value are "sustained";
        shorter events are "short".

    Returns
    -------
    events DataFrame with an added `duration_class` column
    ("sustained" | "short").
    """
    out = events.copy()
    out["duration_class"] = np.where(
        out["duration_hours"] >= sustained_threshold_hours, "sustained", "short"
    )
    return out
