"""
Generate deterministic synthetic 15-minute electricity demand data for
three meters, exercising the analytical functions required by
Implementation Handoff Specification v0.5 Section 44.

Meters:
    B001 - Admin building. Sharp weekday morning+afternoon peaks. Has
           missing intervals and one incomplete day.
    B002 - Academic building. Midday peak, coincident with B001 afternoon
           peak on some days (portfolio peak coincidence test case).
    B003 - Overnight-heavy load (e.g. data closet / continuous process).
           Non-coincident peaks (peaks at night, opposite of B001/B002).

Groups:
    Administration = [B001]
    Academic = [B002]
    Campus_A (hierarchical parent) = Administration + Academic
    Overnight = [B003]  (overlaps: B003 also in "AllMeters" portfolio)

Includes: weekday/weekend behavior, temperature variation with a
piecewise (change-point) response, one incomplete day (partial-day
outage on B001), missing intervals (short gaps for interpolation
testing), and one unusual/outlier day (unseasonal spike on B002).

Deterministic via fixed random seed (42). Interval-ending timestamps.
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

START_DATE = pd.Timestamp("2025-06-01 00:15:00")  # first interval-ending ts
N_DAYS = 21  # 3 weeks: enough for weekday/weekend + seasonality checks
INTERVAL_MIN = 15
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MIN  # 96

timestamps = pd.date_range(START_DATE, periods=N_DAYS * INTERVALS_PER_DAY, freq="15min")


def hourly_temperature(ts_index: pd.DatetimeIndex) -> np.ndarray:
    """Diurnal temperature curve (deg F) with mild day-to-day random walk."""
    day_frac = (ts_index.hour * 60 + ts_index.minute) / (24 * 60)
    base = 68 + 15 * np.sin(2 * np.pi * (day_frac - 0.30))  # trough ~5am, peak ~3pm
    day_index = (ts_index - ts_index[0]).days
    daily_offset = np.repeat(
        rng.normal(0, 4, size=N_DAYS), INTERVALS_PER_DAY
    )[: len(ts_index)]
    return base + daily_offset


temperature_f = hourly_temperature(timestamps)


def demand_profile_B001(ts_index: pd.DatetimeIndex, temp: np.ndarray) -> np.ndarray:
    """Admin building: sharp AM (8-9) and PM (16-18) peaks on weekdays only."""
    hour = ts_index.hour + ts_index.minute / 60
    is_weekday = ts_index.dayofweek < 5
    base = 150 + 0.0 * hour
    am_peak = 300 * np.exp(-0.5 * ((hour - 8.5) / 0.6) ** 2)
    pm_peak = 380 * np.exp(-0.5 * ((hour - 17.0) / 0.8) ** 2)
    weekday_load = base + am_peak + pm_peak
    weekend_load = base * 0.35
    demand = np.where(is_weekday, weekday_load, weekend_load)
    # cooling load above 75F balance point (change-point behavior)
    cooling = np.maximum(temp - 75, 0) * 6.0
    demand = demand + cooling
    demand = demand + rng.normal(0, 8, size=len(ts_index))
    return np.clip(demand, 20, None)


def demand_profile_B002(ts_index: pd.DatetimeIndex, temp: np.ndarray) -> np.ndarray:
    """Academic building: broad midday peak (11-15), weekday only, coincident
    with B001's PM peak tail on some days."""
    hour = ts_index.hour + ts_index.minute / 60
    is_weekday = ts_index.dayofweek < 5
    base = 100
    midday = 250 * np.exp(-0.5 * ((hour - 13.0) / 1.6) ** 2)
    weekday_load = base + midday
    weekend_load = base * 0.4
    demand = np.where(is_weekday, weekday_load, weekend_load)
    cooling = np.maximum(temp - 75, 0) * 4.0
    demand = demand + cooling
    demand = demand + rng.normal(0, 6, size=len(ts_index))
    return np.clip(demand, 15, None)


def demand_profile_B003(ts_index: pd.DatetimeIndex, temp: np.ndarray) -> np.ndarray:
    """Overnight-heavy continuous process load: peaks at night (0-4h),
    non-coincident with B001/B002 daytime peaks. No weekday/weekend effect."""
    hour = ts_index.hour + ts_index.minute / 60
    base = 180
    night_peak = 90 * np.exp(-0.5 * ((hour - 2.0) / 1.5) ** 2)
    night_peak_wrap = 90 * np.exp(-0.5 * ((hour - 26.0) / 1.5) ** 2)  # wrap near midnight
    demand = base + night_peak + night_peak_wrap
    demand = demand + rng.normal(0, 5, size=len(ts_index))
    return np.clip(demand, 50, None)


demand_B001 = demand_profile_B001(timestamps, temperature_f)
demand_B002 = demand_profile_B002(timestamps, temperature_f)
demand_B003 = demand_profile_B003(timestamps, temperature_f)

# --- Inject unusual/outlier day: unseasonal spike on B002, day index 10 ---
outlier_day = timestamps[0].normalize() + pd.Timedelta(days=10)
outlier_mask = (timestamps >= outlier_day) & (timestamps < outlier_day + pd.Timedelta(days=1))
demand_B002 = np.asarray(demand_B002).copy()
demand_B002[np.asarray(outlier_mask)] += 400  # large unseasonal all-day spike

df = pd.DataFrame(
    {
        "timestamp": np.tile(timestamps, 3),
        "meter_id": np.repeat(["B001", "B002", "B003"], len(timestamps)),
        "demand_kw": np.concatenate([demand_B001, demand_B002, demand_B003]),
        "temperature_f": np.tile(temperature_f, 3),
    }
)

# --- Inject short missing-data gaps (interpolation test cases) ---
# B001: 3 consecutive missing intervals (45 min gap) on day 3
gap_day = timestamps[0].normalize() + pd.Timedelta(days=3)
gap_start = gap_day + pd.Timedelta(hours=10)
gap_times = pd.date_range(gap_start, periods=3, freq="15min")
drop_mask = (df["meter_id"] == "B001") & (df["timestamp"].isin(gap_times))
df = df[~drop_mask]

# --- Inject incomplete day: B001 outage for back half of day 5 (partial day) ---
outage_day = timestamps[0].normalize() + pd.Timedelta(days=5)
outage_start = outage_day + pd.Timedelta(hours=13)
outage_end = outage_day + pd.Timedelta(days=1)
outage_mask = (
    (df["meter_id"] == "B001")
    & (df["timestamp"] >= outage_start)
    & (df["timestamp"] < outage_end)
)
df = df[~outage_mask]

# --- Duplicate timestamp validation case: one deliberate duplicate on B003 ---
dup_row = df[(df["meter_id"] == "B003")].iloc[[500]].copy()
df = pd.concat([df, dup_row], ignore_index=True)

df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
df["demand_kw"] = df["demand_kw"].round(2)
df["temperature_f"] = df["temperature_f"].round(1)

out_path = "/home/claude/proj/data/synthetic_load_data.csv"
df.to_csv(out_path, index=False)
print(f"wrote {len(df)} rows to {out_path}")
print(df["meter_id"].value_counts())
