"""
Missing-data handling.

Implements Specification Section 7 (Missing Data) and the
observed/interpolated/analysis distinction of Section 3 / Section 47.
Operates per-meter: builds the expected regular timestamp grid at the
detected resolution, reindexes onto it (introducing NaN for missing
intervals), then interpolates according to configuration -- never
overwriting an observed value, and never filling gaps larger than
max_interpolation_intervals.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import schema

_logger = logging.getLogger(__name__)


def _build_expected_grid(ts: pd.Series, interval_minutes: int) -> pd.DatetimeIndex:
    start, end = ts.min(), ts.max()
    return pd.date_range(start, end, freq=f"{interval_minutes}min")


def handle_missing_data(
    df: pd.DataFrame,
    interval_minutes: int,
    missing_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Detect and (optionally) interpolate missing intervals, per meter.

    Parameters
    ----------
    df:
        Canonical observation DataFrame with columns: timestamp
        (datetime64), meter_id, demand_kw (post duplicate-removal).
        May contain multiple meters; processed independently per meter
        so one meter's gaps never borrow from another meter's grid.
    interval_minutes:
        Native interval duration used to build the expected regular grid.
    missing_cfg:
        The data.missing configuration section: interpolation_enabled,
        method ("linear"|"time"), max_interpolation_intervals,
        large_gap_method, forward_fill_enabled, backward_fill_enabled.

    Returns
    -------
    pandas.DataFrame
        One row per expected interval per meter (reindexed onto the
        regular grid spanning each meter's own min..max timestamp).
        Adds columns: observed_demand_kw, interpolated_demand_kw,
        analysis_demand_kw, is_observed, is_interpolated,
        data_quality_flag. Also carries temperature_f through
        (interpolated the same way if present) and reindexed meter_id.

    Algorithm
    ---------
    1. For each meter, build the expected regular timestamp grid from
       that meter's own min/max observed timestamp.
    2. Reindex the meter's observations onto that grid (missing
       intervals become NaN rows).
    3. observed_demand_kw = original demand_kw (untouched).
    4. If interpolation_enabled, run pandas interpolate() with the
       configured method, limited by max_interpolation_intervals so
       gaps larger than the limit remain NaN (per Section 7: "Do not
       silently fill large gaps").
    5. analysis_demand_kw = observed value where present, else the
       interpolated value where available, else remains NaN.
    6. is_observed / is_interpolated / data_quality_flag record status.

    Assumptions
    -----------
    forward_fill/backward_fill are applied only if explicitly enabled,
    and only after interpolation, and are also limited by
    max_interpolation_intervals.
    large_gap_method="profile" is a documented extension point (Section
    7) and is NOT implemented in Stage 1; gaps larger than the max are
    left missing regardless of this setting.

    Edge Cases
    ----------
    A meter with only one observation cannot form a grid larger than
    one row; returned unchanged with is_observed=True for that row.
    """
    interpolation_enabled = missing_cfg.get("interpolation_enabled", True)
    method = missing_cfg.get("method", "linear")
    max_gap = missing_cfg.get("max_interpolation_intervals", 4)
    fwd_fill = missing_cfg.get("forward_fill_enabled", False)
    bwd_fill = missing_cfg.get("backward_fill_enabled", False)

    out_frames = []
    for meter_id, group in df.groupby("meter_id", sort=False):
        group = group.sort_values("timestamp").drop_duplicates(subset="timestamp")
        grid = _build_expected_grid(group["timestamp"], interval_minutes)
        indexed = group.set_index("timestamp").reindex(grid)
        indexed.index.name = "timestamp"
        indexed["meter_id"] = meter_id

        indexed["observed_demand_kw"] = indexed["demand_kw"]
        indexed["is_observed"] = indexed["demand_kw"].notna()

        if interpolation_enabled:
            interp_method = "time" if method == "time" else "linear"
            interpolated_series = indexed["demand_kw"].interpolate(
                method=interp_method, limit=max_gap, limit_area="inside"
            )
            if fwd_fill:
                interpolated_series = interpolated_series.ffill(limit=max_gap)
            if bwd_fill:
                interpolated_series = interpolated_series.bfill(limit=max_gap)
        else:
            interpolated_series = indexed["demand_kw"]

        indexed["interpolated_demand_kw"] = np.where(
            indexed["is_observed"], np.nan, interpolated_series
        )
        indexed["analysis_demand_kw"] = np.where(
            indexed["is_observed"], indexed["observed_demand_kw"], interpolated_series
        )
        indexed["is_interpolated"] = (~indexed["is_observed"]) & interpolated_series.notna()

        indexed["data_quality_flag"] = np.select(
            [indexed["is_observed"], indexed["is_interpolated"]],
            [schema.QUALITY_FLAG_OBSERVED, schema.QUALITY_FLAG_INTERPOLATED],
            default=schema.QUALITY_FLAG_MISSING,
        )

        # temperature: carry through, interpolate same way if present
        if "temperature_f" in indexed.columns and interpolation_enabled:
            indexed["temperature_f"] = indexed["temperature_f"].interpolate(
                method="linear", limit=max_gap, limit_area="inside"
            )

        indexed = indexed.reset_index()
        out_frames.append(indexed)

    result = pd.concat(out_frames, ignore_index=True)
    result = result.drop(columns=["demand_kw"])  # superseded by analysis_demand_kw
    return result


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    """Basic counts + gap-run stats for a single meter's rows."""
    group = group.sort_values("timestamp")
    n = len(group)
    n_observed = int(group["is_observed"].sum())
    n_interpolated = int(group["is_interpolated"].sum())
    n_missing = n - n_observed - n_interpolated

    missing_mask = ~group["is_observed"]
    run_id = missing_mask.ne(missing_mask.shift()).cumsum()
    runs = (
        group.loc[missing_mask]
        .groupby(run_id[missing_mask])["timestamp"]
        .agg(["size", "min", "max"])
    )

    if len(runs):
        worst = runs.loc[runs["size"].idxmax()]
        n_gap_events = len(runs)
        max_gap_intervals = int(worst["size"])
        max_gap_start = worst["min"]
        max_gap_end = worst["max"]
    else:
        n_gap_events = 0
        max_gap_intervals = 0
        max_gap_start = pd.NaT
        max_gap_end = pd.NaT

    def _pct(part: int) -> float:
        return round(100 * part / n, 2) if n else 0.0

    return {
        "n_intervals": n,
        "n_observed": n_observed,
        "pct_observed": _pct(n_observed),
        "n_interpolated": n_interpolated,
        "pct_interpolated": _pct(n_interpolated),
        "n_missing": n_missing,
        "pct_missing": _pct(n_missing),
        "n_gap_events": n_gap_events,
        "max_gap_intervals": max_gap_intervals,
        "max_gap_start": max_gap_start,
        "max_gap_end": max_gap_end,
    }


def summarize_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-meter and portfolio-wide missing-data summary.

    Implements Section 7's per-meter awareness requirement: takes the
    output of handle_missing_data() (one row per expected interval per
    meter, with is_observed/is_interpolated/data_quality_flag already
    set) and reduces it to one summary row per meter, plus a final
    "PORTFOLIO" row aggregating across all meters.

    Gap-run stats (n_gap_events, max_gap_intervals, max_gap_start/end)
    describe contiguous runs of non-observed intervals -- i.e. the
    original raw gaps, regardless of whether interpolation later
    filled them. The PORTFOLIO row sums per-meter counts and reports
    the single worst gap across all meters, rather than merging rows
    across meter boundaries (which would create false cross-meter
    "gaps").

    Returns
    -------
    pandas.DataFrame
        Columns: meter_id, n_intervals, n_observed, pct_observed,
        n_interpolated, pct_interpolated, n_missing (still missing
        after interpolation), pct_missing, n_gap_events,
        max_gap_intervals, max_gap_start, max_gap_end.
    """
    rows = [
        {"meter_id": meter_id, **_summarize_group(group)}
        for meter_id, group in df.groupby("meter_id", sort=False)
    ]
    summary = pd.DataFrame(rows)

    total_intervals = int(summary["n_intervals"].sum())
    total_observed = int(summary["n_observed"].sum())
    total_interpolated = int(summary["n_interpolated"].sum())
    total_missing = int(summary["n_missing"].sum())

    def _pct(part: int) -> float:
        return round(100 * part / total_intervals, 2) if total_intervals else 0.0

    if len(summary):
        worst = summary.loc[summary["max_gap_intervals"].idxmax()]
        max_gap_intervals = int(worst["max_gap_intervals"])
        max_gap_start = worst["max_gap_start"]
        max_gap_end = worst["max_gap_end"]
    else:
        max_gap_intervals = 0
        max_gap_start = pd.NaT
        max_gap_end = pd.NaT

    portfolio_row = pd.DataFrame([{
        "meter_id": "PORTFOLIO",
        "n_intervals": total_intervals,
        "n_observed": total_observed,
        "pct_observed": _pct(total_observed),
        "n_interpolated": total_interpolated,
        "pct_interpolated": _pct(total_interpolated),
        "n_missing": total_missing,
        "pct_missing": _pct(total_missing),
        "n_gap_events": int(summary["n_gap_events"].sum()),
        "max_gap_intervals": max_gap_intervals,
        "max_gap_start": max_gap_start,
        "max_gap_end": max_gap_end,
    }])

    return pd.concat([summary, portfolio_row], ignore_index=True)


def missing_intervals_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level detail of every non-observed interval, per meter.

    Returns the timestamp and resulting data_quality_flag
    ("interpolated" or "missing") for every interval where
    is_observed is False -- the per-interval record behind the counts
    in summarize_missing_data(), for drill-down, export, or follow-up
    action (e.g. contacting a meter owner about a specific outage
    window).

    Returns
    -------
    pandas.DataFrame
        Columns: meter_id, timestamp, data_quality_flag. Sorted by
        meter then timestamp.
    """
    detail = df.loc[~df["is_observed"], ["meter_id", "timestamp", "data_quality_flag"]]
    return detail.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


def log_missing_data_summary(
    summary: pd.DataFrame, logger: logging.Logger | None = None
) -> None:
    """
    Emit the per-meter + portfolio missing-data summary via the
    standard logging module (one INFO record per row).

    Uses Python's stdlib `logging` rather than print() so the report
    is captured consistently whether the pipeline runs interactively
    in a notebook or as a script, and can be routed by the caller's
    own logging configuration (file handler, log aggregator, etc.).

    Parameters
    ----------
    summary:
        Output of summarize_missing_data().
    logger:
        Logger to emit to; defaults to this module's logger.
    """
    log = logger or _logger
    for _, row in summary.iterrows():
        log.info(
            "data_quality meter=%s intervals=%d observed=%d (%.2f%%) "
            "interpolated=%d (%.2f%%) missing=%d (%.2f%%) gap_events=%d "
            "max_gap=%d intervals [%s .. %s]",
            row["meter_id"],
            row["n_intervals"],
            row["n_observed"],
            row["pct_observed"],
            row["n_interpolated"],
            row["pct_interpolated"],
            row["n_missing"],
            row["pct_missing"],
            row["n_gap_events"],
            row["max_gap_intervals"],
            row["max_gap_start"],
            row["max_gap_end"],
        )
