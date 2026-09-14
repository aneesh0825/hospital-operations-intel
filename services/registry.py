"""Registry loading, validation, and patient encounter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from config.constants import MODEL_FEATURES, REQUIRED_REGISTRY_COLUMNS


@dataclass(frozen=True)
class RegistryValidation:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


def load_registry(path: Path) -> pd.DataFrame:
    """Load the scored registry, raising a concise error when it is unavailable."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Registry file was not found: {path}") from error
    except (OSError, pd.errors.ParserError) as error:
        raise RuntimeError(f"Registry could not be loaded from {path}: {error}") from error


def validate_registry(registry: pd.DataFrame) -> RegistryValidation:
    """Report structural and history-window issues without mutating registry data."""
    missing = sorted(REQUIRED_REGISTRY_COLUMNS.difference(registry.columns))
    if missing:
        return RegistryValidation(errors=(f"Missing required columns: {', '.join(missing)}.",))

    warnings: list[str] = []
    probability_ranges = {
        "readmission_probability": (0, 1),
        "readmission_probability_percent": (0, 100),
    }
    for column, (minimum, maximum) in probability_ranges.items():
        values = pd.to_numeric(registry[column], errors="coerce")
        invalid = values.isna() | ~values.between(minimum, maximum)
        if invalid.any():
            warnings.append(
                f"{int(invalid.sum()):,} row(s) have invalid values in {column}; expected {minimum}–{maximum}."
            )

    windows = registry[[
        "admissions_last_30_days", "admissions_last_90_days",
        "admissions_last_365_days", "previous_inpatient_admissions",
    ]].apply(pd.to_numeric, errors="coerce")
    inconsistent = (
        (windows["admissions_last_30_days"] > windows["admissions_last_90_days"])
        | (windows["admissions_last_90_days"] > windows["admissions_last_365_days"])
        | (windows["admissions_last_365_days"] > windows["previous_inpatient_admissions"])
    )
    if inconsistent.any():
        warnings.append(
            f"{int(inconsistent.sum()):,} row(s) have inconsistent nested admission windows."
        )
    return RegistryValidation(warnings=tuple(warnings))


def short_patient_id(patient_id: object) -> str:
    return f"P-{str(patient_id)[-6:].upper()}"


def short_encounter_id(encounter_id: object) -> str:
    return f"E-{str(encounter_id)[-6:].upper()}"


def condition_summary(row: Mapping[str, object]) -> str:
    labels = (
        ("diabetes", "Diabetes"),
        ("hypertension", "Hypertension"),
        ("kidney_disease", "Kidney Disease"),
    )
    flagged = [label for key, label in labels if row.get(key, 0) == 1]
    return ", ".join(flagged) if flagged else "None flagged"


def row_to_model_input(row: Mapping[str, object]) -> dict[str, object]:
    return {feature: row[feature] for feature in MODEL_FEATURES}


def ordered_patient_history(registry: pd.DataFrame, patient_id: object) -> pd.DataFrame:
    """Order encounters by available cumulative-utilization proxy, not calendar time."""
    history = registry.loc[registry["patient_id"] == patient_id].copy()
    history = history.sort_values(
        ["previous_inpatient_admissions", "readmission_probability"],
        ascending=[True, True],
    ).reset_index(drop=True)
    history["Encounter Sequence"] = range(1, len(history) + 1)
    return history


COMPARISON_FIELDS = (
    ("readmission_probability_percent", "Modeled readmission risk", "%"),
    ("admissions_last_30_days", "Admissions in last 30 days", ""),
    ("admissions_last_90_days", "Admissions in last 90 days", ""),
    ("admissions_last_365_days", "Admissions in last 365 days", ""),
    ("prior_readmissions_last_365_days", "Prior readmissions", ""),
    ("days_since_last_inpatient_admission", "Days since last inpatient admission", "days"),
    ("condition_count", "Condition count", ""),
    ("medication_count", "Medication count", ""),
    ("procedure_count", "Procedure count", ""),
)


def encounter_comparison(history: pd.DataFrame, encounter_id: object) -> pd.DataFrame | None:
    """Compare an encounter with its immediate predecessor in proxy encounter order."""
    matches = history.index[history["encounter_id"] == encounter_id].tolist()
    if not matches or matches[0] == 0:
        return None
    current = history.iloc[matches[0]]
    previous = history.iloc[matches[0] - 1]
    rows = []
    for key, label, unit in COMPARISON_FIELDS:
        current_value = float(current[key])
        previous_value = float(previous[key])
        rows.append({
            "Metric": label,
            "Previous": previous_value,
            "Current": current_value,
            "Change": current_value - previous_value,
            "Unit": unit,
        })
    return pd.DataFrame(rows)
