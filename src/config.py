"""
Configuration loading and validation.

Implements Specification Section 35 (Configuration Requirements) and
Section 36 (Configuration Validation) for the Stage 1 scope: input
mapping, time resolution, missing-data policy, meters/groups, calendar,
validation strictness.

Configuration is TOML. No expression evaluation of any kind occurs when
reading the config file (Section 30's "no arbitrary Python execution"
principle is treated as a project-wide rule, not just for the search
engine).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(Exception):
    """Raised when configuration is structurally invalid. Section 36."""


def load_configuration(config_path: str | Path) -> dict[str, Any]:
    """
    Load a TOML configuration file into a plain dict.

    Parameters
    ----------
    config_path:
        Path to the .toml configuration file.

    Returns
    -------
    dict
        Parsed configuration. No validation is performed here; call
        validate_configuration() separately (Section 36 requires
        validation to occur before any data processing).

    Edge Cases
    ----------
    Raises FileNotFoundError if the path does not exist.
    Raises tomllib.TOMLDecodeError if the file is not valid TOML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


@dataclass
class ValidationFinding:
    severity: str  # INFO | WARNING | ERROR
    message: str
    section: str


def validate_configuration(config: dict[str, Any]) -> list[ValidationFinding]:
    """
    Validate configuration structure before any data is loaded.

    Purpose
    -------
    Catch structurally invalid configuration early per Section 36
    (missing required input column mapping, duplicate meter IDs, invalid
    time interval, group references to nonexistent meters, etc.)

    Returns
    -------
    list[ValidationFinding]
        Findings with severity INFO/WARNING/ERROR. Caller decides whether
        to abort; if any ERROR-severity finding exists, processing must
        not proceed (Section 36: "Errors must be clear and actionable").

    Algorithm
    ---------
    Runs a fixed sequence of structural checks. Does not attempt to
    validate against the actual data file contents (that happens in
    validate_input_data, Section 37) -- only the configuration's
    internal consistency.
    """
    findings: list[ValidationFinding] = []

    def err(msg: str, section: str) -> None:
        findings.append(ValidationFinding(severity="ERROR", message=msg, section=section))

    def warn(msg: str, section: str) -> None:
        findings.append(ValidationFinding(severity="WARNING", message=msg, section=section))

    # --- data.input ---
    data_input = config.get("data", {}).get("input", {})
    if not data_input.get("file_path"):
        err("data.input.file_path is required", "data.input")

    # --- column mapping: required canonical fields must be mapped ---
    mapping = config.get("data", {}).get("column_mapping", {})
    for required in ("timestamp", "meter_id", "demand_kw"):
        if required not in mapping or not mapping[required]:
            err(f"data.column_mapping.{required} is required", "data.column_mapping")

    # --- time resolution ---
    resolution = config.get("data", {}).get("time", {}).get("resolution", "auto")
    valid_resolutions = {"auto", "15min", "30min", "60min"}
    if resolution not in valid_resolutions:
        # allow other explicit "<n>min" values but must be well-formed
        if not (resolution.endswith("min") and resolution[:-3].isdigit()):
            err(
                f"data.time.resolution '{resolution}' is not a recognized value",
                "data.time",
            )

    # --- external temperature (optional) ---
    ext_temp = config.get("data", {}).get("external_temperature", {})
    if ext_temp.get("file_path"):
        ext_mapping = ext_temp.get(
            "column_mapping", {"timestamp": "timestamp", "temperature_f": "temperature_f"}
        )
        for required in ("timestamp", "temperature_f"):
            if required not in ext_mapping or not ext_mapping[required]:
                err(
                    f"data.external_temperature.column_mapping.{required} is required "
                    "when data.external_temperature.file_path is set",
                    "data.external_temperature",
                )
        tolerance = ext_temp.get("join_tolerance_minutes")
        if tolerance is not None and (not isinstance(tolerance, (int, float)) or tolerance < 0):
            err(
                "data.external_temperature.join_tolerance_minutes must be a non-negative number",
                "data.external_temperature",
            )

    # --- missing data ---
    missing = config.get("data", {}).get("missing", {})
    max_gap = missing.get("max_interpolation_intervals")
    if max_gap is not None and (not isinstance(max_gap, int) or max_gap < 0):
        err(
            "data.missing.max_interpolation_intervals must be a non-negative integer",
            "data.missing",
        )

    # --- meters: duplicate IDs ---
    meters = config.get("meters", [])
    meter_ids = [m.get("meter_id") for m in meters]
    seen = set()
    for mid in meter_ids:
        if mid is None:
            err("A [[meters]] entry is missing meter_id", "meters")
            continue
        if mid in seen:
            err(f"Duplicate meter_id in [[meters]]: {mid}", "meters")
        seen.add(mid)

    # --- meter groups: reference validity, including hierarchical child_groups ---
    groups = config.get("meter_groups", [])
    group_names = {g.get("name") for g in groups if g.get("name")}
    known_meter_ids = set(meter_ids)
    for g in groups:
        name = g.get("name")
        if not name:
            err("A [[meter_groups]] entry is missing name", "meter_groups")
            continue
        for m in g.get("meters", []):
            if m not in known_meter_ids:
                err(
                    f"Group '{name}' references unknown meter_id '{m}'",
                    "meter_groups",
                )
        for child in g.get("child_groups", []):
            if child not in group_names:
                err(
                    f"Group '{name}' references unknown child_group '{child}'",
                    "meter_groups",
                )
            if child == name:
                err(f"Group '{name}' lists itself as a child_group", "meter_groups")

    # --- cycle detection in group hierarchy ---
    child_map = {g.get("name"): g.get("child_groups", []) for g in groups if g.get("name")}

    def has_cycle(start: str) -> bool:
        visited: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                return True
            visited.add(node)
            stack.extend(child_map.get(node, []))
        return False

    for name in child_map:
        if has_cycle(name):
            err(f"Cyclic group hierarchy detected involving '{name}'", "meter_groups")
            break

    # --- validation strictness section itself ---
    validation_cfg = config.get("validation", {})
    for key in ("negative_demand", "duplicate_timestamp", "nonnumeric_demand"):
        val = validation_cfg.get(key, "error")
        if val not in ("error", "warning"):
            err(f"validation.{key} must be 'error' or 'warning', got '{val}'", "validation")

    if not meters:
        warn("No [[meters]] defined in configuration", "meters")

    return findings


def raise_if_errors(findings: list[ValidationFinding]) -> None:
    """Raise ConfigurationError if any ERROR-severity finding is present."""
    errors = [f for f in findings if f.severity == "ERROR"]
    if errors:
        lines = [f"[{f.severity}] {f.section}: {f.message}" for f in errors]
        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(lines)
        )
