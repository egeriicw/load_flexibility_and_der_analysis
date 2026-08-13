"""
Stage 1 test suite (Specification Section 43, items applicable to
schema/config/ingestion/time/missing-data/entities/profiles scope).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg_mod
from src import ingestion
from src import timeproc
from src import missing
from src import entities
from src import profiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "example_configuration.toml"


@pytest.fixture(scope="module")
def config():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    findings = cfg_mod.validate_configuration(c)
    cfg_mod.raise_if_errors(findings)
    return c


@pytest.fixture(scope="module")
def canonical_df(config):
    raw = ingestion.load_input_data(config, base_dir=PROJECT_ROOT)
    mapped = ingestion.map_to_canonical_schema(raw, config)
    mapped["timestamp"] = pd.to_datetime(mapped["timestamp"])
    return mapped


# ---------------------------------------------------------------- config --
def test_config_loads():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    assert "meters" in c
    assert len(c["meters"]) == 3


def test_config_validation_passes_on_example(config):
    findings = cfg_mod.validate_configuration(config)
    errors = [f for f in findings if f.severity == "ERROR"]
    assert errors == []


def test_config_rejects_unknown_meter_in_group():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    c["meter_groups"].append({"name": "Bad", "meters": ["NOPE"]})
    findings = cfg_mod.validate_configuration(c)
    assert any("unknown meter_id" in f.message for f in findings)


def test_config_rejects_duplicate_meter_id():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    c["meters"].append({"meter_id": "B001"})
    findings = cfg_mod.validate_configuration(c)
    assert any("Duplicate meter_id" in f.message for f in findings)


def test_config_rejects_cyclic_group_hierarchy():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    c["meter_groups"].append({"name": "X", "meters": [], "child_groups": ["Y"]})
    c["meter_groups"].append({"name": "Y", "meters": [], "child_groups": ["X"]})
    findings = cfg_mod.validate_configuration(c)
    assert any("Cyclic" in f.message for f in findings)


def test_config_missing_required_mapping_is_error():
    c = cfg_mod.load_configuration(CONFIG_PATH)
    del c["data"]["column_mapping"]["demand_kw"]
    findings = cfg_mod.validate_configuration(c)
    assert any("demand_kw" in f.message and f.severity == "ERROR" for f in findings)


# ------------------------------------------------------------- ingestion --
def test_load_and_map(canonical_df):
    for col in ("timestamp", "meter_id", "demand_kw", "temperature_f"):
        assert col in canonical_df.columns


def test_validate_input_data_reports_negative_demand(config, canonical_df):
    bad = canonical_df.copy()
    bad.loc[bad.index[0], "demand_kw"] = -50
    findings = ingestion.validate_input_data(bad, config)
    assert any("negative" in f.message.lower() for f in findings)


def test_validate_input_data_reports_nonnumeric(config, canonical_df):
    bad = canonical_df.copy()
    bad["demand_kw"] = bad["demand_kw"].astype(object)
    bad.loc[bad.index[0], "demand_kw"] = "not_a_number"
    findings = ingestion.validate_input_data(bad, config)
    assert any("nonnumeric" in f.message.lower() for f in findings)


def test_validate_input_data_reports_duplicate_timestamp(config, canonical_df):
    bad = pd.concat([canonical_df, canonical_df.iloc[[0]]], ignore_index=True)
    findings = ingestion.validate_input_data(bad, config)
    assert any("duplicate" in f.message.lower() for f in findings)


def test_synthetic_data_has_no_negative_or_nonnumeric(config, canonical_df):
    # sanity check on the shipped synthetic dataset itself (dupe expected: 1)
    findings = ingestion.validate_input_data(canonical_df, config)
    msgs = [f.message for f in findings]
    assert not any("negative" in m.lower() for m in msgs)
    assert not any("nonnumeric" in m.lower() for m in msgs)
    assert any("1 duplicate" in m for m in msgs)  # generator injects exactly 1


# --------------------------------------------------------------- timeproc --
def test_detect_resolution_15min(canonical_df):
    ts = pd.to_datetime(canonical_df["timestamp"])
    result = timeproc.detect_time_resolution(ts, meter_id=canonical_df["meter_id"])
    assert result["expected_interval_minutes"] == 15
    assert result["is_mixed_resolution"] is False


def test_energy_calculation_15min():
    demand = pd.Series([100.0])
    energy = timeproc.calculate_interval_energy(demand, 15)
    assert energy.iloc[0] == pytest.approx(25.0)


def test_energy_calculation_30min():
    demand = pd.Series([100.0])
    energy = timeproc.calculate_interval_energy(demand, 30)
    assert energy.iloc[0] == pytest.approx(50.0)


def test_energy_calculation_60min():
    demand = pd.Series([100.0])
    energy = timeproc.calculate_interval_energy(demand, 60)
    assert energy.iloc[0] == pytest.approx(100.0)


def test_calendar_features_weekday_weekend_holiday():
    ts = pd.Series(pd.to_datetime(["2025-06-02", "2025-06-07", "2025-06-19"]))  # Mon, Sat, holiday(Thu)
    cal = timeproc.build_calendar_features(ts, holidays=["2025-06-19"])
    assert cal["day_type"].tolist() == ["weekday", "weekend", "holiday"]
    assert cal["is_weekday"].tolist() == [True, False, True]  # holiday is still a Thursday


# ---------------------------------------------------------------- missing --
def test_handle_missing_data_interpolates_short_gap(config, canonical_df):
    resolution_min = 15
    df = handle_and_get(config, canonical_df, resolution_min)
    b001 = df[df["meter_id"] == "B001"]
    assert b001["is_interpolated"].sum() >= 3  # 3-interval gap injected by generator


def test_handle_missing_data_never_overwrites_observed(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    observed = df[df["is_observed"]]
    assert (observed["analysis_demand_kw"] == observed["observed_demand_kw"]).all()


def test_handle_missing_data_flags_incomplete_day_gap(config, canonical_df):
    # the outage on day 5 (13:00 onward) exceeds max_interpolation_intervals(4)
    df = handle_and_get(config, canonical_df, 15)
    b001 = df[df["meter_id"] == "B001"]
    still_missing = b001[b001["data_quality_flag"] == "missing"]
    assert len(still_missing) > 4  # large gap correctly left unfilled


def handle_and_get(config, canonical_df, resolution_min):
    return missing.handle_missing_data(
        canonical_df.rename(columns={"timestamp": "timestamp"}),
        resolution_min,
        config["data"]["missing"],
    )


def test_summarize_missing_data_per_meter_and_portfolio(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    summary = missing.summarize_missing_data(df)

    assert set(summary["meter_id"]) == {"B001", "B002", "B003", "PORTFOLIO"}

    b001 = summary[summary["meter_id"] == "B001"].iloc[0]
    assert b001["n_interpolated"] >= 3
    assert b001["n_gap_events"] >= 1
    assert b001["max_gap_intervals"] >= b001["n_missing"]  # incomplete-day gap dominates

    portfolio = summary[summary["meter_id"] == "PORTFOLIO"].iloc[0]
    per_meter = summary[summary["meter_id"] != "PORTFOLIO"]
    assert portfolio["n_intervals"] == per_meter["n_intervals"].sum()
    assert portfolio["n_observed"] == per_meter["n_observed"].sum()
    assert portfolio["n_interpolated"] == per_meter["n_interpolated"].sum()
    assert portfolio["n_missing"] == per_meter["n_missing"].sum()
    assert portfolio["max_gap_intervals"] == per_meter["max_gap_intervals"].max()


def test_missing_intervals_detail_lists_only_non_observed_rows(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    detail = missing.missing_intervals_detail(df)

    assert set(detail.columns) == {"meter_id", "timestamp", "data_quality_flag"}
    assert (detail["data_quality_flag"] != "observed").all()
    assert len(detail) == (df["data_quality_flag"] != "observed").sum()
    assert list(detail["meter_id"]) == sorted(detail["meter_id"])


def test_log_missing_data_summary_emits_one_record_per_row(config, canonical_df, caplog):
    df = handle_and_get(config, canonical_df, 15)
    summary = missing.summarize_missing_data(df)

    with caplog.at_level("INFO", logger="src.missing"):
        missing.log_missing_data_summary(summary)

    records = [r for r in caplog.records if r.name == "src.missing"]
    assert len(records) == len(summary)
    assert any("PORTFOLIO" in r.message for r in records)


# --------------------------------------------------------------- entities --
def test_build_meter_groups_flat_and_overlap(config):
    groups = entities.build_meter_groups(config)
    assert groups["Administration"] == ["B001"]
    assert groups["Academic"] == ["B002"]


def test_build_meter_groups_hierarchical(config):
    groups = entities.build_meter_groups(config)
    assert groups["Campus_A"] == ["B001", "B002"]
    assert groups["AllMeters"] == ["B001", "B002", "B003"]


def test_portfolio_meters(config):
    portfolio = entities.build_portfolio_meters(config)
    assert portfolio == ["B001", "B002", "B003"]


def test_aggregate_entity_load_sums_not_averages():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:15"] * 3),
            "meter_id": ["A", "B", "C"],
            "analysis_demand_kw": [500.0, 300.0, 200.0],
        }
    )
    agg = entities.aggregate_entity_load(df, ["A", "B", "C"])
    assert agg["demand_kw"].iloc[0] == pytest.approx(1000.0)


def test_aggregate_entity_load_reports_partial_coverage():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:15"] * 2),
            "meter_id": ["A", "B"],
            "analysis_demand_kw": [500.0, np.nan],
        }
    )
    agg = entities.aggregate_entity_load(df, ["A", "B"])
    assert agg["n_meters_reporting"].iloc[0] == 1
    assert agg["demand_kw"].iloc[0] == pytest.approx(500.0)


# --------------------------------------------------------------- profiles --
def test_daily_profile_interval_counts_by_resolution():
    assert profiles.expected_intervals_per_day(15) == 96
    assert profiles.expected_intervals_per_day(30) == 48
    assert profiles.expected_intervals_per_day(60) == 24


def test_construct_daily_profiles_flags_incomplete_day(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    b001 = df[df["meter_id"] == "B001"].rename(columns={"analysis_demand_kw": "demand_kw"})
    daily = profiles.construct_daily_profiles(b001, 15, entity_id="B001")
    outage_date = (pd.Timestamp("2025-06-01") + pd.Timedelta(days=5)).date()
    day_rows = daily[daily["date"] == outage_date]
    assert day_rows["is_complete_day"].iloc[0] == False  # noqa: E712


def test_calculate_daily_features_load_factor_bounds(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    b002 = df[df["meter_id"] == "B002"].rename(columns={"analysis_demand_kw": "demand_kw"})
    daily = profiles.construct_daily_profiles(b002, 15, entity_id="B002")
    feats = profiles.calculate_daily_features(daily, 15)
    complete = feats[feats["is_complete_day"]]
    assert (complete["load_factor"] <= 1.0001).all()
    assert (complete["load_factor"] > 0).all()


def test_normalize_daily_profiles_peak_is_one(config, canonical_df):
    df = handle_and_get(config, canonical_df, 15)
    b003 = df[df["meter_id"] == "B003"].rename(columns={"analysis_demand_kw": "demand_kw"})
    daily = profiles.construct_daily_profiles(b003, 15, entity_id="B003")
    norm = profiles.normalize_daily_profiles(daily)
    complete_dates = daily[daily["is_complete_day"]]["date"].unique()
    for d in complete_dates[:3]:
        day_rows = norm[norm["date"] == d]
        assert day_rows["normalized_demand"].max() == pytest.approx(1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
