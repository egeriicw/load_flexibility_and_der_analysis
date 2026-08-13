"""
Time processing: resolution detection, calendar features, interval energy.

Implements Specification Section 2.1/2.2 (demand/energy semantics),
Section 6 (Time Resolution), Section 9 (Calendar Model).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SEASON_BY_MONTH_DEFAULT = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def detect_time_resolution(timestamps: pd.Series, meter_id: pd.Series | None = None) -> dict:
    """
    Detect native interval resolution from timestamp differences.

    Purpose
    -------
    Determine the expected interval (minutes) per Section 6, and flag
    irregular intervals, duplicate timestamps, and mixed resolution.

    Parameters
    ----------
    timestamps:
        Parsed datetime Series (interval-ending timestamps).
    meter_id:
        Optional Series aligned with timestamps; if provided, detection
        is performed per-meter and results aggregated, since resolution
        should not be judged across different meters' interleaved rows.

    Returns
    -------
    dict with keys:
        expected_interval_minutes: int | None
        is_mixed_resolution: bool
        n_irregular_gaps: int
        n_duplicate_timestamps: int
        per_meter: dict[meter_id, int]  (only if meter_id supplied)

    Algorithm
    ---------
    For each meter's timestamp series (or the whole series if meter_id
    is None), compute consecutive differences, take the mode as the
    expected interval, and count differences that deviate from it.
    """
    result = {
        "expected_interval_minutes": None,
        "is_mixed_resolution": False,
        "n_irregular_gaps": 0,
        "n_duplicate_timestamps": 0,
        "per_meter": {},
    }

    def per_series_mode_minutes(ts: pd.Series) -> tuple[int | None, int, int]:
        ts_sorted = ts.sort_values()
        diffs = ts_sorted.diff().dropna()
        n_dupes = int((diffs == pd.Timedelta(0)).sum())
        diffs_nonzero = diffs[diffs > pd.Timedelta(0)]
        if diffs_nonzero.empty:
            return None, 0, n_dupes
        minutes = (diffs_nonzero.dt.total_seconds() / 60).round().astype(int)
        mode_val = int(minutes.mode().iloc[0])
        n_irregular = int((minutes != mode_val).sum())
        return mode_val, n_irregular, n_dupes

    if meter_id is not None:
        modes = []
        for mid, group in timestamps.groupby(meter_id):
            mode_val, n_irr, n_dup = per_series_mode_minutes(group)
            result["per_meter"][mid] = mode_val
            result["n_irregular_gaps"] += n_irr
            result["n_duplicate_timestamps"] += n_dup
            if mode_val is not None:
                modes.append(mode_val)
        unique_modes = set(modes)
        result["is_mixed_resolution"] = len(unique_modes) > 1
        result["expected_interval_minutes"] = (
            int(pd.Series(modes).mode().iloc[0]) if modes else None
        )
    else:
        mode_val, n_irr, n_dup = per_series_mode_minutes(timestamps)
        result["expected_interval_minutes"] = mode_val
        result["n_irregular_gaps"] = n_irr
        result["n_duplicate_timestamps"] = n_dup

    return result


def calculate_interval_energy(demand_kw: pd.Series, interval_minutes: int) -> pd.Series:
    """
    Calculate interval energy from average interval demand.

    Parameters
    ----------
    demand_kw:
        Average electricity demand during each measurement interval, kW.
    interval_minutes:
        Duration of each measurement interval in minutes.

    Returns
    -------
    pandas.Series
        Energy represented by each interval in kWh.

    Algorithm
    ---------
        energy_kwh = demand_kw * interval_minutes / 60

    Assumptions
    -----------
    Demand represents average demand over the interval (Section 2.1).
    Timestamp represents interval-ending time; not used in this calc.

    Missing-data Behavior
    ----------------------
    Missing demand remains missing (NaN propagates). No imputation here.
    """
    return demand_kw * interval_minutes / 60.0


def build_calendar_features(
    timestamps: pd.Series,
    holidays: list[str] | None = None,
    season_by_month: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Derive calendar features from interval-ending timestamps.

    Purpose
    -------
    Implements Section 9 (Calendar Model): date/year/month/day/day-of-week
    /weekday-weekend/season/day_type (weekday|weekend|holiday).

    Parameters
    ----------
    timestamps:
        Parsed datetime Series.
    holidays:
        List of ISO date strings (YYYY-MM-DD) treated as holidays,
        overriding weekday/weekend day_type.
    season_by_month:
        Optional override of month->season mapping; defaults to
        Northern Hemisphere meteorological seasons.

    Returns
    -------
    pandas.DataFrame
        One row per input timestamp with calendar columns. Custom day
        types (Section 9) are an extension point not populated here.

    Assumptions
    -----------
    is_weekday/is_weekend reflect calendar weekday regardless of
    holiday status; day_type is the single authoritative classification
    used downstream and holiday takes precedence over weekday/weekend.
    """
    season_map = season_by_month or SEASON_BY_MONTH_DEFAULT
    holiday_dates = set(pd.to_datetime(holidays).date) if holidays else set()

    dates = timestamps.dt.date
    dow = timestamps.dt.dayofweek  # Monday=0
    is_weekday = dow < 5

    day_type = np.where(
        dates.isin(holiday_dates) if hasattr(dates, "isin") else pd.Series(dates).isin(holiday_dates),
        "holiday",
        np.where(is_weekday, "weekday", "weekend"),
    )

    out = pd.DataFrame(
        {
            "date": dates,
            "year": timestamps.dt.year,
            "month": timestamps.dt.month,
            "day": timestamps.dt.day,
            "day_of_year": timestamps.dt.dayofyear,
            "hour": timestamps.dt.hour,
            "minute": timestamps.dt.minute,
            "day_of_week": dow,
            "day_name": timestamps.dt.day_name(),
            "is_weekday": is_weekday,
            "is_weekend": ~is_weekday,
            "season": timestamps.dt.month.map(season_map),
            "day_type": day_type,
        }
    )
    return out
