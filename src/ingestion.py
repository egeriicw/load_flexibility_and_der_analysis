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


def load_external_temperature_data(
    config: dict[str, Any], base_dir: str | Path = "."
) -> pd.DataFrame | None:
    """
    Load a standalone temperature time series from a separate file.

    Purpose
    -------
    Site temperature is sometimes supplied by a separate weather-station
    export rather than as a column in the load-data file. This is
    controlled by the optional data.external_temperature config section.

    Parameters
    ----------
    config:
        Parsed configuration. Reads data.external_temperature.file_path,
        .file_format, .column_mapping, .datetime.format.
    base_dir:
        Directory that file_path is resolved relative to.

    Returns
    -------
    pandas.DataFrame with columns [timestamp, temperature_f], sorted by
    timestamp, or None if data.external_temperature.file_path is not
    set (external temperature is an opt-in source).

    Assumptions
    -----------
    Only file_format = "csv" is implemented, matching load_input_data.
    """
    ext_cfg = config.get("data", {}).get("external_temperature", {})
    file_path_str = ext_cfg.get("file_path")
    if not file_path_str:
        return None

    file_format = ext_cfg.get("file_format", "csv")
    file_path = Path(base_dir) / file_path_str
    if not file_path.exists():
        raise FileNotFoundError(f"External temperature file not found: {file_path}")
    if file_format != "csv":
        raise NotImplementedError(
            f"file_format '{file_format}' is not supported for external temperature data"
        )

    raw = pd.read_csv(file_path)

    mapping = ext_cfg.get(
        "column_mapping", {"timestamp": "timestamp", "temperature_f": "temperature_f"}
    )
    rename_map = {source_col: canonical for canonical, source_col in mapping.items()}
    df = raw.rename(columns=rename_map)

    for required in ("timestamp", "temperature_f"):
        if required not in df.columns:
            raise ValueError(
                f"External temperature file missing required column '{required}' "
                "after applying data.external_temperature.column_mapping"
            )

    dt_fmt = ext_cfg.get("datetime", {}).get("format", "auto")
    if dt_fmt and dt_fmt != "auto":
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=dt_fmt)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")
    df = (
        df[["timestamp", "temperature_f"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


def apply_external_temperature(
    canonical_df: pd.DataFrame, config: dict[str, Any], base_dir: str | Path = "."
) -> pd.DataFrame:
    """
    Merge an external temperature series into canonical_df, if configured.

    Purpose
    -------
    Site temperature is a single time series shared across meters, so it
    is joined on timestamp alone (not meter_id). Called whenever
    data.external_temperature.file_path is set -- covers both the case
    where the input file has no temperature_f column at all, and the
    case where an external source is configured to supplement or
    override a partially-populated one.

    Parameters
    ----------
    canonical_df:
        Canonically-mapped DataFrame with a parsed (datetime64)
        'timestamp' column.
    config:
        Parsed configuration. Reads data.external_temperature.
        override_existing (bool, default False) and
        .join_tolerance_minutes (nearest-match window; unset/0 means no
        tolerance limit).
    base_dir:
        Directory that data.external_temperature.file_path is resolved
        relative to.

    Returns
    -------
    pandas.DataFrame
        Copy of canonical_df with temperature_f populated from the
        external source. If canonical_df has no temperature_f column,
        one is added. If external temperature is not configured
        (load_external_temperature_data returns None), canonical_df is
        returned unchanged.

    Behavior
    --------
    Default (override_existing=false): external values fill only rows
    where temperature_f is missing/absent; existing observed values are
    kept. override_existing=true: external values take precedence
    wherever a match is found, falling back to the existing value only
    where no external match exists (e.g. outside join_tolerance_minutes).
    """
    external_temp = load_external_temperature_data(config, base_dir=base_dir)
    if external_temp is None:
        return canonical_df

    if "timestamp" not in canonical_df.columns or not pd.api.types.is_datetime64_any_dtype(
        canonical_df["timestamp"]
    ):
        raise ValueError(
            "canonical_df must have a parsed datetime 'timestamp' column before "
            "merging external temperature"
        )

    ext_cfg = config.get("data", {}).get("external_temperature", {})
    override = bool(ext_cfg.get("override_existing", False))
    tolerance_minutes = ext_cfg.get("join_tolerance_minutes")
    tolerance = pd.Timedelta(minutes=tolerance_minutes) if tolerance_minutes else None

    df = canonical_df.copy()
    order = df["timestamp"].argsort(kind="mergesort")
    left = df.iloc[order][["timestamp"]]

    merged = pd.merge_asof(
        left,
        external_temp.rename(columns={"temperature_f": "_external_temperature_f"}),
        on="timestamp",
        direction="nearest",
        tolerance=tolerance,
    )
    external_values = pd.Series(merged["_external_temperature_f"].values, index=left.index)
    external_values = external_values.reindex(df.index)

    if "temperature_f" not in df.columns:
        df["temperature_f"] = external_values
    elif override:
        df["temperature_f"] = external_values.where(external_values.notna(), df["temperature_f"])
    else:
        df["temperature_f"] = df["temperature_f"].where(df["temperature_f"].notna(), external_values)

    return df


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
