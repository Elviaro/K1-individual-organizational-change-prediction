from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("person_step_2_transitions.xlsx")
OUTPUT_FILE = Path("person_step_4_marker_transition_audit.xlsx")

MARKERS = [
    "vp",
    "svp",
    "evp",
    "sevp",
    "dir",
    "sdir",
    "md",
    "smd",
    "se",
    "vc",
    "svc",
    "president",
    "board",
    "ceo",
    "cxo",
    "primary",
    "support",
    "bu",
]

# Years used in the previous rolling evaluation. They are used here only
# to check whether each transition type occurs often enough every year.
ROLLING_TEST_YEARS = [2019, 2020, 2021, 2022, 2023]

# These thresholds are diagnostic rules, not statistical guarantees.
# "recommended" means that a marker has enough observed events to justify
# a separate predictive experiment. "exploratory" means that a model may
# be attempted, but its estimates are likely to be unstable.
RECOMMENDED_TOTAL_EVENTS = 500
RECOMMENDED_EVENTS_PER_TEST_YEAR = 20
EXPLORATORY_TOTAL_EVENTS = 200
EXPLORATORY_EVENTS_PER_TEST_YEAR = 10


# ============================================================
# LOADING AND VALIDATION
# ============================================================

def load_transitions(path: Path) -> pd.DataFrame:
    """Load the person-level adjacent-year transition table."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}"
        )

    workbook = pd.ExcelFile(path)

    preferred_sheets = [
        "person_transitions",
        "transitions",
    ]
    sheet_name = next(
        (
            sheet
            for sheet in preferred_sheets
            if sheet in workbook.sheet_names
        ),
        workbook.sheet_names[0],
    )

    print(
        f"Loading {path.name}, sheet '{sheet_name}'...",
        flush=True,
    )

    data = pd.read_excel(path, sheet_name=sheet_name)

    print(
        f"Loaded {len(data):,} transitions and "
        f"{len(data.columns):,} columns.",
        flush=True,
    )

    return data


def validate_data(data: pd.DataFrame) -> None:
    """Validate marker pairs, years and binary marker values."""

    required = [
        "year",
        *MARKERS,
        *[f"next_{marker}" for marker in MARKERS],
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    marker_columns = [
        *MARKERS,
        *[f"next_{marker}" for marker in MARKERS],
    ]

    invalid_rows = pd.Series(False, index=data.index)

    for column in marker_columns:
        invalid_rows |= ~data[column].isin([0, 1])

    if invalid_rows.any():
        raise ValueError(
            f"Found {int(invalid_rows.sum()):,} rows with "
            "missing or non-binary marker values."
        )

    if "next_year" in data.columns:
        invalid_gap = (
            pd.to_numeric(data["next_year"])
            - pd.to_numeric(data["year"])
            != 1
        )

        if invalid_gap.any():
            raise ValueError(
                f"Found {int(invalid_gap.sum()):,} non-adjacent "
                "year pairs."
            )

    print("Validation completed successfully.", flush=True)


# ============================================================
# TRANSITION CALCULATIONS
# ============================================================

def transition_counts(
    current: pd.Series,
    following: pd.Series,
) -> dict:
    """Count the four possible binary transitions."""

    current = current.astype("int8")
    following = following.astype("int8")

    n_00 = int(((current == 0) & (following == 0)).sum())
    n_01 = int(((current == 0) & (following == 1)).sum())
    n_10 = int(((current == 1) & (following == 0)).sum())
    n_11 = int(((current == 1) & (following == 1)).sum())

    return {
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
    }


def build_yearly_table(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate marker transition frequencies by target year."""

    if "next_year" in data.columns:
        target_year = pd.to_numeric(data["next_year"]).astype(int)
    else:
        target_year = pd.to_numeric(data["year"]).astype(int) + 1

    rows = []

    for marker in MARKERS:
        following_column = f"next_{marker}"

        for year in sorted(target_year.unique()):
            mask = target_year == year
            counts = transition_counts(
                data.loc[mask, marker],
                data.loc[mask, following_column],
            )

            current_zero = counts["n_00"] + counts["n_01"]
            current_one = counts["n_10"] + counts["n_11"]
            changes = counts["n_01"] + counts["n_10"]
            n_pairs = int(mask.sum())

            rows.append({
                "marker": marker,
                "target_year": int(year),
                "n_pairs": n_pairs,
                **counts,
                "onset_eligible_n": current_zero,
                "onset_events_n": counts["n_01"],
                "onset_rate": (
                    counts["n_01"] / current_zero
                    if current_zero
                    else np.nan
                ),
                "loss_eligible_n": current_one,
                "loss_events_n": counts["n_10"],
                "loss_rate": (
                    counts["n_10"] / current_one
                    if current_one
                    else np.nan
                ),
                "change_events_n": changes,
                "change_rate": (
                    changes / n_pairs
                    if n_pairs
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def readiness_label(
    total_events: int,
    minimum_year_events: int,
) -> str:
    """Assign a transparent diagnostic modelling-readiness label."""

    if (
        total_events >= RECOMMENDED_TOTAL_EVENTS
        and minimum_year_events
        >= RECOMMENDED_EVENTS_PER_TEST_YEAR
    ):
        return "recommended"

    if (
        total_events >= EXPLORATORY_TOTAL_EVENTS
        and minimum_year_events
        >= EXPLORATORY_EVENTS_PER_TEST_YEAR
    ):
        return "exploratory"

    return "insufficient_events"


def build_summary(
    data: pd.DataFrame,
    yearly: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main marker-level transition and readiness summary."""

    rows = []

    rolling = yearly[
        yearly["target_year"].isin(ROLLING_TEST_YEARS)
    ]

    for marker in MARKERS:
        counts = transition_counts(
            data[marker],
            data[f"next_{marker}"],
        )

        current_zero = counts["n_00"] + counts["n_01"]
        current_one = counts["n_10"] + counts["n_11"]
        changes = counts["n_01"] + counts["n_10"]
        n_pairs = len(data)

        marker_yearly = rolling[rolling["marker"] == marker]

        yearly_onset = (
            marker_yearly
            .set_index("target_year")["onset_events_n"]
            .reindex(ROLLING_TEST_YEARS, fill_value=0)
        )
        yearly_loss = (
            marker_yearly
            .set_index("target_year")["loss_events_n"]
            .reindex(ROLLING_TEST_YEARS, fill_value=0)
        )

        min_onset_year = int(yearly_onset.min())
        min_loss_year = int(yearly_loss.min())

        rows.append({
            "marker": marker,
            "n_pairs": n_pairs,
            **counts,
            "current_0_n": current_zero,
            "current_1_n": current_one,
            "current_1_share": (
                current_one / n_pairs
                if n_pairs
                else np.nan
            ),
            "onset_events_n": counts["n_01"],
            "onset_rate": (
                counts["n_01"] / current_zero
                if current_zero
                else np.nan
            ),
            "minimum_onset_events_2019_2023": min_onset_year,
            "onset_model_readiness": readiness_label(
                counts["n_01"],
                min_onset_year,
            ),
            "loss_events_n": counts["n_10"],
            "loss_rate": (
                counts["n_10"] / current_one
                if current_one
                else np.nan
            ),
            "minimum_loss_events_2019_2023": min_loss_year,
            "loss_model_readiness": readiness_label(
                counts["n_10"],
                min_loss_year,
            ),
            "all_change_events_n": changes,
            "all_change_rate": (
                changes / n_pairs
                if n_pairs
                else np.nan
            ),
            "persistence_baseline_accuracy": (
                (counts["n_00"] + counts["n_11"]) / n_pairs
                if n_pairs
                else np.nan
            ),
        })

    summary = pd.DataFrame(rows)

    readiness_order = {
        "recommended": 0,
        "exploratory": 1,
        "insufficient_events": 2,
    }

    summary["_sort"] = summary[
        "onset_model_readiness"
    ].map(readiness_order)

    summary = (
        summary
        .sort_values(
            ["_sort", "all_change_events_n"],
            ascending=[True, False],
        )
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    return summary


def build_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Document the population and linkage scope used by this audit."""

    rows = [
        {
            "metric": "adjacent_person_year_pairs",
            "value": len(data),
            "interpretation": (
                "Rows read from the verified Step 2 transition table."
            ),
        },
        {
            "metric": "markers",
            "value": len(MARKERS),
            "interpretation": (
                "Binary structural markers evaluated separately."
            ),
        },
    ]

    if "person_key" in data.columns:
        rows.append({
            "metric": "unique_person_keys",
            "value": int(data["person_key"].nunique()),
            "interpretation": (
                "person_key already includes GV_KEY, so the analysis "
                "tracks a person within the same organization."
            ),
        })

    if {"GV_KEY", "next_GV_KEY"}.issubset(data.columns):
        same_company = (
            data["GV_KEY"].astype(str)
            == data["next_GV_KEY"].astype(str)
        )

        rows.extend([
            {
                "metric": "same_company_pairs",
                "value": int(same_company.sum()),
                "interpretation": (
                    "Pairs with identical current and next GV_KEY."
                ),
            },
            {
                "metric": "company_mismatch_pairs",
                "value": int((~same_company).sum()),
                "interpretation": (
                    "Should normally be zero because person_key "
                    "contains the organization key."
                ),
            },
        ])

    return pd.DataFrame(rows)


def build_method_table() -> pd.DataFrame:
    """Describe the statistical meaning of the output columns."""

    return pd.DataFrame([
        {
            "item": "0 -> 0",
            "definition": (
                "Marker absent in year t and absent in year t+1."
            ),
        },
        {
            "item": "0 -> 1 (onset)",
            "definition": (
                "Marker absent in year t and appears in year t+1."
            ),
        },
        {
            "item": "1 -> 0 (loss)",
            "definition": (
                "Marker present in year t and disappears in year t+1."
            ),
        },
        {
            "item": "1 -> 1",
            "definition": (
                "Marker present in year t and remains in year t+1."
            ),
        },
        {
            "item": "onset model population",
            "definition": (
                "Only rows with marker=0 in year t; target is 0 -> 1."
            ),
        },
        {
            "item": "loss model population",
            "definition": (
                "Only rows with marker=1 in year t; target is 1 -> 0."
            ),
        },
        {
            "item": "persistence baseline",
            "definition": (
                "Predict that the marker remains unchanged. This is the "
                "minimum meaningful comparator for next-state accuracy."
            ),
        },
        {
            "item": "recommended",
            "definition": (
                f"At least {RECOMMENDED_TOTAL_EVENTS} total events and "
                f"at least {RECOMMENDED_EVENTS_PER_TEST_YEAR} events in "
                "each rolling test year 2019-2023."
            ),
        },
        {
            "item": "exploratory",
            "definition": (
                f"At least {EXPLORATORY_TOTAL_EVENTS} total events and "
                f"at least {EXPLORATORY_EVENTS_PER_TEST_YEAR} events in "
                "each rolling test year 2019-2023."
            ),
        },
        {
            "item": "insufficient_events",
            "definition": (
                "Too few observed events for a stable separate rolling "
                "model under the diagnostic thresholds above."
            ),
        },
    ])


# ============================================================
# OUTPUT
# ============================================================

def save_results(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    quality: pd.DataFrame,
    method: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save all diagnostic tables to a single Excel workbook."""

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(
            writer,
            sheet_name="marker_summary",
            index=False,
        )
        yearly.to_excel(
            writer,
            sheet_name="yearly_transitions",
            index=False,
        )
        quality.to_excel(
            writer,
            sheet_name="data_quality",
            index=False,
        )
        method.to_excel(
            writer,
            sheet_name="method",
            index=False,
        )

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                width = min(
                    max(
                        len(str(cell.value))
                        if cell.value is not None
                        else 0
                        for cell in column_cells
                    ) + 2,
                    70,
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = width

    print(
        f"Result saved to: {output_path.resolve()}",
        flush=True,
    )


def main() -> None:
    started = perf_counter()

    print(
        "STEP 4.1/4: Loading verified person transitions...",
        flush=True,
    )
    data = load_transitions(INPUT_FILE)
    validate_data(data)

    print(
        "STEP 4.2/4: Counting 0->0, 0->1, 1->0 and 1->1 "
        "for every marker...",
        flush=True,
    )
    yearly = build_yearly_table(data)
    summary = build_summary(data, yearly)

    print(
        "STEP 4.3/4: Assessing event sufficiency for rolling models...",
        flush=True,
    )
    quality = build_quality_summary(data)
    method = build_method_table()

    print(
        "STEP 4.4/4: Saving the audit workbook...",
        flush=True,
    )
    save_results(
        summary,
        yearly,
        quality,
        method,
        OUTPUT_FILE,
    )

    display_columns = [
        "marker",
        "n_01",
        "onset_rate",
        "onset_model_readiness",
        "n_10",
        "loss_rate",
        "loss_model_readiness",
        "all_change_rate",
        "persistence_baseline_accuracy",
    ]

    print("\nMARKER TRANSITION SUMMARY", flush=True)
    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    elapsed = perf_counter() - started

    print(
        f"\nSTEP 4 COMPLETE. Total time: {elapsed / 60:.1f} min.",
        flush=True,
    )


if __name__ == "__main__":
    main()
