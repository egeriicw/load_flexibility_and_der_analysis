"""
Data ingestion and canonical mapping.

Implements Specification Section 37 (Data Validation) and the ingestion
portion of Section 3 (Canonical Data Model) / Section 38 (function
architecture): load_input_data, validate_input_data, map_to_canonical_schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ValidationFinding
from . import schema


def load_input_data(config: dict[str, Any], base_dir: str | Path = ".") -> pd.DataFrame:
    """
    Load the raw input file specified in configuration.

    Parameters
    ----------
    config:
        Parsed configuration (see config.load_configuration).
    base_dir:
        Directory that data.input.file_path is resolved relative to.

    Returns
    -------
    pandas.DataFrame
        Raw data exactly as read from the source file, with source
        column names unchanged (no canonical mapping applied yet).

    Assumptions
    -----------
    Only file_format = "csv" is implemented in Stage 1. Other formats
    raise NotImplementedError (extension point).
    """
    input_cfg = config["data"]["input"]
    file_format = input_cfg.get("file_format", "csv")
    file_path = Path(base_dir) / input_cfg["file_path"]

    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found: {file_path}")

    if file_format == "csv":
        return pd.read_csv(file_path)
    raise NotImplementedError(f"file_format '{file_format}' is not supported in Stage 1")


def map_to_canonical_schema(raw_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Rename source columns to canonical field names using data.column_mapping.

    Purpose
    -------
    Decouple the analytical engine from arbitrary source file column
    names (Section 3, Section 4 item "map arbitrary source column names
    into a canonical internal data model").

    Parameters
    ----------
    raw_df:
        DataFrame as returned by load_input_data.
    config:
        Parsed configuration containing data.column_mapping.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns renamed to canonical names. Only mapped
        columns are kept plus any unmapped columns are dropped silently
        is NOT done -- unmapped extra source columns are retained
        unchanged so no source information is discarded.

    Assumptions
    -----------
    data.column_mapping values must exist as columns in raw_df; this is
    checked in validate_input_data, not here.
    """
    mapping = config["data"]["column_mapping"]
    # invert: source_col -> canonical_name
    rename_map = {source_col: canonical for canonical, source_col in mapping.items()}
    return raw_df.rename(columns=rename_map)


def validate_input_data(
    df: pd.DataFrame, config: dict[str, Any]
) -> list[ValidationFinding]:
    """
    Validate a canonically-mapped DataFrame per Section 37.

    Checks: required columns present, datetime parses, duplicate
    meter_id/timestamp combinations, negative demand, nonnumeric demand,
    missing meter identifiers, missing dates.

    Parameters
    ----------
    df:
        Canonically-mapped DataFrame (post map_to_canonical_schema),
        BEFORE datetime parsing/dtype coercion.
    config:
        Parsed configuration (uses validation.* severity settings).

    Returns
    -------
    list[ValidationFinding]

    Missing-data Behavior
    ----------------------
    This function does not fill or drop anything; it only reports.
    """
    findings: list[ValidationFinding] = []
    validation_cfg = config.get("validation", {})

    def add(condition: bool, msg: str, severity_key: str, default_severity: str, section: str):
        if condition:
            severity = validation_cfg.get(severity_key, default_severity)
            findings.append(ValidationFinding(severity.upper(), msg, section))

    # --- required columns ---
    for required in schema.REQUIRED_FIELDS:
        if required not in df.columns:
            findings.append(
                ValidationFinding(schema.SEVERITY_ERROR, f"Required column '{required}' missing after mapping", "ingestion")
            )
    if any(f.severity == schema.SEVERITY_ERROR for f in findings):
        return findings  # cannot proceed further checks without required columns

    # --- datetime parsing ---
    dt_fmt = config.get("data", {}).get("datetime", {}).get("format", "auto")
    try:
        if dt_fmt and dt_fmt != "auto":
            parsed = pd.to_datetime(df["timestamp"], format=dt_fmt, errors="coerce")
        else:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_dates = parsed.isna().sum()
    add(n_bad_dates > 0, f"{n_bad_dates} timestamp value(s) failed to parse", "nonnumeric_demand", schema.SEVERITY_ERROR, "ingestion")

    # --- demand numeric ---
    demand_numeric = pd.to_numeric(df["demand_kw"], errors="coerce")
    n_nonnumeric = demand_numeric.isna().sum() - df["demand_kw"].isna().sum()
    add(
        n_nonnumeric > 0,
        f"{n_nonnumeric} demand_kw value(s) are nonnumeric",
        "nonnumeric_demand",
        schema.SEVERITY_ERROR,
        "ingestion",
    )

    # --- negative demand (Section 2.3: unsupported in this version) ---
    n_negative = (demand_numeric < 0).sum()
    add(
        n_negative > 0,
        f"{n_negative} negative demand_kw value(s) detected; negative demand is not supported in this version",
        "negative_demand",
        schema.SEVERITY_ERROR,
        "ingestion",
    )

    # --- missing meter identifiers ---
    n_missing_meter = df["meter_id"].isna().sum()
    findings.extend(
        [ValidationFinding(schema.SEVERITY_ERROR, f"{n_missing_meter} row(s) missing meter_id", "ingestion")]
        if n_missing_meter > 0
        else []
    )

    # --- duplicate meter_id/timestamp combinations ---
    combo = pd.DataFrame({"meter_id": df["meter_id"], "timestamp": parsed})
    n_dupes = combo.duplicated(subset=["meter_id", "timestamp"]).sum()
    add(
        n_dupes > 0,
        f"{n_dupes} duplicate (meter_id, timestamp) record(s) detected",
        "duplicate_timestamp",
        schema.SEVERITY_ERROR,
        "ingestion",
    )

    return findings
