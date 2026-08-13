"""
Meter/entity/group/portfolio construction.

Implements Specification Section 4 (Physical Meters and Entities) and
Section 5 (Meter Grouping): flat groups, overlapping groups, hierarchical
groups, and portfolio (sum of all meters).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def build_meter_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """
    Resolve configured meter groups (including hierarchical child_groups)
    into flat lists of physical meter_ids per group.

    Parameters
    ----------
    config:
        Parsed configuration containing [[meter_groups]] entries, each
        with `name`, `meters` (direct members), and optional
        `child_groups` (names of other groups whose resolved members are
        included -- enables hierarchy, e.g. Campus_A -> Administration +
        Academic).

    Returns
    -------
    dict[str, list[str]]
        group_name -> sorted list of unique physical meter_ids
        (de-duplicated; a meter reachable via multiple paths appears
        once). Overlap across groups is preserved -- the SAME meter_id
        may appear in multiple groups' lists.

    Algorithm
    ---------
    Recursive resolution with memoization; cycle protection assumes
    validate_configuration() has already rejected cyclic hierarchies.

    Assumptions
    -----------
    Group names are unique (enforced upstream). A group's own `meters`
    list and its child groups' resolved members are unioned.
    """
    groups_cfg = {g["name"]: g for g in config.get("meter_groups", [])}
    resolved: dict[str, list[str]] = {}

    def resolve(name: str, _stack: tuple[str, ...] = ()) -> list[str]:
        if name in resolved:
            return resolved[name]
        if name in _stack:
            raise ValueError(f"Cyclic group hierarchy detected at '{name}'")
        g = groups_cfg[name]
        members = set(g.get("meters", []))
        for child in g.get("child_groups", []):
            members |= set(resolve(child, _stack + (name,)))
        resolved[name] = sorted(members)
        return resolved[name]

    for name in groups_cfg:
        resolve(name)

    return resolved


def build_portfolio_meters(config: dict[str, Any]) -> list[str]:
    """
    Determine the set of physical meters comprising the portfolio.

    Purpose
    -------
    Section 5: "The portfolio consists of all included physical meters
    unless explicitly configured otherwise."

    Returns
    -------
    list[str]
        Sorted meter_ids in the portfolio (all configured meters minus
        portfolio.excluded_meters, unless portfolio.include_all_meters
        is false, in which case portfolio.excluded_meters is still
        honored against the full meter list -- Stage 1 does not support
        an explicit alternate portfolio meter list; that is an
        extension point).
    """
    all_meters = [m["meter_id"] for m in config.get("meters", [])]
    excluded = set(config.get("portfolio", {}).get("excluded_meters", []))
    return sorted(set(all_meters) - excluded)


def aggregate_entity_load(
    df: pd.DataFrame, meter_ids: list[str], demand_col: str = "analysis_demand_kw"
) -> pd.DataFrame:
    """
    Aggregate per-interval demand across a set of meters into one entity
    load series by SUMMATION (never averaging).

    Parameters
    ----------
    df:
        Canonical observation DataFrame containing at minimum
        `timestamp`, `meter_id`, and `demand_col`.
    meter_ids:
        Physical meters comprising the entity (group or portfolio).
    demand_col:
        Column to sum. Defaults to analysis_demand_kw (observed value
        where available, else interpolated) so entity aggregates use the
        same value a single-meter analysis would use.

    Returns
    -------
    pandas.DataFrame
        Columns: timestamp, demand_kw (summed), n_meters_reporting
        (count of non-null contributing meters at that timestamp -- an
        entity interval where fewer than len(meter_ids) meters reported
        is itself a data-quality signal, surfaced here rather than
        hidden).

    Algorithm
    ---------
        entity_demand_kw(t) = sum over m in meter_ids of demand_col(m, t)

    Non-contributing (NaN) meters are excluded from the sum via
    pandas' default skipna behavior; n_meters_reporting records how many
    meters actually contributed at each timestamp so a partial-coverage
    aggregate is never silently indistinguishable from a full one.

    Edge Cases
    ----------
    If meter_ids is empty, returns an empty DataFrame with the expected
    columns.
    """
    if not meter_ids:
        return pd.DataFrame(columns=["timestamp", "demand_kw", "n_meters_reporting"])

    subset = df[df["meter_id"].isin(meter_ids)]
    grouped = subset.groupby("timestamp")[demand_col]
    out = grouped.sum(min_count=1).rename("demand_kw").to_frame()
    out["n_meters_reporting"] = grouped.apply(lambda s: s.notna().sum())
    out = out.reset_index()
    return out
