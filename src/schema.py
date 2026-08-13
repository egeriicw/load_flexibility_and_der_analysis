"""
Canonical data model definitions.

Implements Specification Section 3 (Canonical Data Model). This module
contains no logic -- only field name constants and dtype expectations --
so every other module references field names from a single source of
truth rather than hard-coded strings.
"""

# Required canonical input fields (must exist after mapping)
REQUIRED_FIELDS = ["timestamp", "meter_id", "demand_kw"]

# Optional canonical input fields
OPTIONAL_FIELDS = ["temperature_f"]

# Fields derived during time processing (Section 6-9)
CALENDAR_FIELDS = [
    "date",
    "year",
    "month",
    "day",
    "day_of_year",
    "hour",
    "minute",
    "day_of_week",
    "day_name",
    "is_weekday",
    "is_weekend",
    "season",
    "day_type",  # weekday | weekend | holiday | custom_day_type
]

# Fields derived during interval/energy processing (Section 2, 6)
INTERVAL_FIELDS = [
    "interval_minutes",
    "interval_hours",
    "energy_kwh",
]

# Fields derived during missing-data handling (Section 3, 7, 47)
QUALITY_FIELDS = [
    "observed_demand_kw",
    "interpolated_demand_kw",
    "analysis_demand_kw",
    "is_observed",
    "is_interpolated",
    "data_quality_flag",
]

# All fields the canonical observation table carries after Stage 1 processing
CANONICAL_OBSERVATION_FIELDS = (
    REQUIRED_FIELDS
    + OPTIONAL_FIELDS
    + CALENDAR_FIELDS
    + INTERVAL_FIELDS
    + QUALITY_FIELDS
)

# Valid data_quality_flag values
QUALITY_FLAG_OBSERVED = "observed"
QUALITY_FLAG_INTERPOLATED = "interpolated"
QUALITY_FLAG_MISSING = "missing"

# Severity levels for validation findings (Section 46)
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_ERROR = "ERROR"
