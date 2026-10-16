from pathlib import Path
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("person_step_2_transitions.xlsx")
OUTPUT_FILE = Path("person_step_5_marker_models_result.xlsx")

RANDOM_STATE = 42
TEST_YEARS = [2019, 2020, 2021, 2022, 2023]

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

# Primary targets selected before model fitting from the Step 4 event audit.
# Five markers have enough appearance and disappearance events. VP and SVP
# are additionally modelled only for disappearance.
TARGETS = [
    ("bu", "onset"),
    ("bu", "loss"),
    ("cxo", "onset"),
    ("cxo", "loss"),
    ("primary", "onset"),
    ("primary", "loss"),
    ("support", "onset"),
    ("support", "loss"),
    ("evp", "onset"),
    ("evp", "loss"),
    ("svp", "loss"),
    ("vp", "loss"),
]

# One pre-specified model is used for every target and year. This avoids
# selecting a separate model configuration after observing test results.
MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}


# ============================================================
# PROGRESS
# ============================================================

def progress(message: str) -> None:
    """Print a timestamped message immediately."""

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


# ============================================================
# LOADING AND VALIDATION
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load the verified person-transition table."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}"
        )

    workbook = pd.ExcelFile(path)
    preferred = ["person_transitions", "transitions"]
    sheet_name = next(
        (name for name in preferred if name in workbook.sheet_names),
        workbook.sheet_names[0],
    )

    progress(
        f"Loading {path.name}, sheet '{sheet_name}'..."
    )

    data = pd.read_excel(path, sheet_name=sheet_name)

    progress(
        f"Loaded {len(data):,} transitions and "
        f"{len(data.columns):,} columns."
    )

    return data


def validate_data(data: pd.DataFrame) -> None:
    """Validate identifiers, years and binary marker pairs."""

    required = [
        "person_key",
        "company",
        "GV_KEY",
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

    data["year"] = pd.to_numeric(data["year"]).astype(int)

    if "next_year" not in data.columns:
        data["next_year"] = data["year"] + 1
    else:
        data["next_year"] = pd.to_numeric(
            data["next_year"]
        ).astype(int)

    invalid_gap = data["next_year"] - data["year"] != 1

    if invalid_gap.any():
        raise ValueError(
            f"Found {int(invalid_gap.sum()):,} non-adjacent pairs."
        )

    marker_columns = [
        *MARKERS,
        *[f"next_{marker}" for marker in MARKERS],
    ]

    invalid = pd.Series(False, index=data.index)

    for column in marker_columns:
        invalid |= ~data[column].isin([0, 1])
        data[column] = data[column].astype("int8")

    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum()):,} rows with "
            "missing or non-binary marker values."
        )

    duplicate_keys = data.duplicated(
        ["person_key", "year"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError(
            f"Found {int(duplicate_keys.sum()):,} duplicate "
            "person_key-year rows."
        )

    progress("Input validation completed successfully.")


# ============================================================
# LEAKAGE-SAFE FEATURE ENGINEERING
# ============================================================

def prepare_common_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features that do not depend on a particular target."""

    result = data.copy()
    result = result.sort_values(
        ["person_key", "year"]
    ).reset_index(drop=True)

    result["current_marker_count"] = result[MARKERS].sum(axis=1)
    result["multiple_current_markers"] = (
        result["current_marker_count"] > 1
    ).astype("int8")

    result["any_change_current_transition"] = (
        result[MARKERS].to_numpy()
        != result[[f"next_{m}" for m in MARKERS]].to_numpy()
    ).any(axis=1).astype("int8")

    grouped_person = result.groupby(
        "person_key",
        sort=False,
    )

    result["prior_transition_count"] = grouped_person.cumcount()
    result["first_observed_year"] = grouped_person["year"].transform(
        "min"
    )
    result["years_since_first_observation"] = (
        result["year"] - result["first_observed_year"]
    )

    result["prior_any_change_count"] = (
        grouped_person["any_change_current_transition"]
        .cumsum()
        - result["any_change_current_transition"]
    )
    result["prior_any_change_rate"] = np.where(
        result["prior_transition_count"] > 0,
        result["prior_any_change_count"]
        / result["prior_transition_count"],
        0.0,
    )
    result["previous_transition_changed"] = (
        grouped_person["any_change_current_transition"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    # Verify that the previous record is truly the immediately preceding
    # adjacent-year transition. If not, history continuity is set to zero.
    previous_year = grouped_person["year"].shift(1)
    result["previous_transition_is_adjacent"] = (
        previous_year.eq(result["year"] - 1)
    ).astype("int8")
    result.loc[
        result["previous_transition_is_adjacent"] == 0,
        "previous_transition_changed",
    ] = 0

    # Current company-year composition. These values use only marker states
    # observed at time t, never next-year outcomes.
    company_groups = result.groupby(
        ["GV_KEY", "year"],
        sort=False,
    )

    result["company_tmt_size"] = company_groups[
        "person_key"
    ].transform("size")

    for marker in MARKERS:
        count_column = f"company_{marker}_count"
        share_column = f"company_{marker}_share"

        result[count_column] = company_groups[marker].transform("sum")
        result[share_column] = (
            result[count_column] / result["company_tmt_size"]
        )

    return result


def build_target_data(
    common: pd.DataFrame,
    marker: str,
    direction: str,
) -> tuple[pd.DataFrame, list[str], str]:
    """Create one onset or loss modelling population and its features."""

    if direction not in {"onset", "loss"}:
        raise ValueError(f"Unknown direction: {direction}")

    current_required = 0 if direction == "onset" else 1
    population = common[
        common[marker] == current_required
    ].copy()

    target_column = "target_event"

    if direction == "onset":
        population[target_column] = population[
            f"next_{marker}"
        ].astype("int8")
    else:
        population[target_column] = (
            1 - population[f"next_{marker}"]
        ).astype("int8")

    # Event history for the person is shifted by one transition, so the
    # outcome being predicted never enters its own predictors.
    common_event = pd.Series(
        np.where(
            direction == "onset",
            (
                (common[marker] == 0)
                & (common[f"next_{marker}"] == 1)
            ),
            (
                (common[marker] == 1)
                & (common[f"next_{marker}"] == 0)
            ),
        ).astype("int8"),
        index=common.index,
    )

    common_history = common[[
        "person_key",
        "year",
    ]].copy()
    common_history["target_event_for_history"] = common_event
    common_history = common_history.sort_values(
        ["person_key", "year"]
    )
    history_group = common_history.groupby(
        "person_key",
        sort=False,
    )
    common_history["prior_target_event_count"] = (
        history_group["target_event_for_history"].cumsum()
        - common_history["target_event_for_history"]
    )
    common_history["previous_target_event"] = (
        history_group["target_event_for_history"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    population = population.merge(
        common_history[
            [
                "person_key",
                "year",
                "prior_target_event_count",
                "previous_target_event",
            ]
        ],
        on=["person_key", "year"],
        how="left",
        validate="one_to_one",
    )

    population["prior_target_event_rate"] = np.where(
        population["prior_transition_count"] > 0,
        population["prior_target_event_count"]
        / population["prior_transition_count"],
        0.0,
    )

    # Company event history is calculated from completed transitions whose
    # target year equals the current prediction year. Thus, for t -> t+1,
    # only outcomes observed no later than year t are used.
    event_source = common[[
        "GV_KEY",
        "year",
        "next_year",
        marker,
        f"next_{marker}",
    ]].copy()

    if direction == "onset":
        event_source = event_source[event_source[marker] == 0]
        event_source["historical_event"] = event_source[
            f"next_{marker}"
        ]
    else:
        event_source = event_source[event_source[marker] == 1]
        event_source["historical_event"] = (
            1 - event_source[f"next_{marker}"]
        )

    company_history = (
        event_source
        .groupby(["GV_KEY", "next_year"], as_index=False)
        .agg(
            company_prior_eligible=("historical_event", "size"),
            company_prior_target_events=("historical_event", "sum"),
        )
        .rename(columns={"next_year": "year"})
    )
    company_history["company_prior_target_rate"] = (
        company_history["company_prior_target_events"]
        / company_history["company_prior_eligible"]
    )

    population = population.merge(
        company_history,
        on=["GV_KEY", "year"],
        how="left",
        validate="many_to_one",
    )

    company_history_columns = [
        "company_prior_eligible",
        "company_prior_target_events",
        "company_prior_target_rate",
    ]
    population[company_history_columns] = population[
        company_history_columns
    ].fillna(0)

    other_markers = [m for m in MARKERS if m != marker]

    feature_columns = [
        *other_markers,
        "current_marker_count",
        "multiple_current_markers",
        "years_since_first_observation",
        "prior_transition_count",
        "prior_any_change_count",
        "prior_any_change_rate",
        "previous_transition_changed",
        "previous_transition_is_adjacent",
        "prior_target_event_count",
        "prior_target_event_rate",
        "previous_target_event",
        "company_tmt_size",
        "company_prior_eligible",
        "company_prior_target_events",
        "company_prior_target_rate",
    ]

    for company_marker in MARKERS:
        feature_columns.extend([
            f"company_{company_marker}_count",
            f"company_{company_marker}_share",
        ])

    population[feature_columns] = population[
        feature_columns
    ].replace([np.inf, -np.inf], np.nan).fillna(0)

    return population, feature_columns, target_column


# ============================================================
# METRICS
# ============================================================

def safe_roc_auc(y_true, probability) -> float:
    """Calculate ROC-AUC when both classes are present."""

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    return float(roc_auc_score(y_true, probability))


def risk_group_metrics(
    y_true,
    probability,
    share: float,
) -> dict:
    """Calculate precision, lift and captured events in a top-risk group."""

    y_array = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    n_selected = max(1, int(np.ceil(len(y_array) * share)))
    selected = np.argsort(-probability, kind="stable")[:n_selected]

    selected_positive = int(y_array[selected].sum())
    total_positive = int(y_array.sum())
    prevalence = float(y_array.mean())
    precision = selected_positive / n_selected
    lift = precision / prevalence if prevalence > 0 else np.nan
    captured_share = (
        selected_positive / total_positive
        if total_positive > 0
        else np.nan
    )

    label = int(round(share * 100))

    return {
        f"top_{label}_n": n_selected,
        f"top_{label}_precision": precision,
        f"top_{label}_lift": lift,
        f"top_{label}_event_capture": captured_share,
    }


def complete_metrics(
    y_true,
    probability,
    prediction,
) -> dict:
    """Calculate ranking, classification and risk-concentration metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, probability))

    result = {
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
        "positive_share": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_over_prevalence": (
            pr_auc / prevalence
            if prevalence > 0
            else np.nan
        ),
        "roc_auc": safe_roc_auc(y_true, probability),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    for share in [0.05, 0.10, 0.20]:
        result.update(
            risk_group_metrics(y_true, probability, share)
        )

    return result


def select_threshold(y_true, probability) -> tuple[float, pd.DataFrame]:
    """Select the validation threshold that maximizes F1."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probability,
    )

    if len(thresholds) == 0:
        return 0.5, pd.DataFrame()

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    best = table.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    return float(best["threshold"]), table


# ============================================================
# TEMPORAL MODELLING
# ============================================================

def split_fold(
    data: pd.DataFrame,
    target_column: str,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create expanding train, one-year validation and one-year test sets."""

    validation_year = test_year - 1

    train = data[data["next_year"] < validation_year].copy()
    validation = data[
        data["next_year"] == validation_year
    ].copy()
    test = data[data["next_year"] == test_year].copy()

    for name, part in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if part.empty:
            raise ValueError(
                f"{name} is empty for target year {test_year}."
            )

        if part[target_column].nunique() < 2:
            raise ValueError(
                f"{name} contains only one target class for "
                f"target year {test_year}."
            )

    return train, validation, test


def fit_model(
    data: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> HistGradientBoostingClassifier:
    """Fit the pre-specified class-balanced model."""

    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    weights = compute_sample_weight(
        class_weight="balanced",
        y=data[target_column],
    )

    model.fit(
        data[features],
        data[target_column],
        sample_weight=weights,
    )

    return model


def run_fold(
    data: pd.DataFrame,
    features: list[str],
    target_column: str,
    marker: str,
    direction: str,
    test_year: int,
) -> dict:
    """Tune the threshold on validation and evaluate a hidden test year."""

    started = perf_counter()
    train, validation, test = split_fold(
        data,
        target_column,
        test_year,
    )

    progress(
        f"{marker} {direction}, target year {test_year}: "
        f"train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}."
    )

    validation_model = fit_model(
        train,
        features,
        target_column,
    )
    validation_probability = validation_model.predict_proba(
        validation[features]
    )[:, 1]
    threshold, threshold_table = select_threshold(
        validation[target_column],
        validation_probability,
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    model = fit_model(
        development,
        features,
        target_column,
    )

    probability = model.predict_proba(test[features])[:, 1]
    prediction = (probability >= threshold).astype(int)

    development_prevalence = float(
        development[target_column].mean()
    )
    baseline_probability = np.full(
        len(test),
        development_prevalence,
    )
    baseline_prediction = np.zeros(len(test), dtype=int)

    common = {
        "marker": marker,
        "direction": direction,
        "target_year": test_year,
        "validation_year": test_year - 1,
        "train_min_target_year": int(train["next_year"].min()),
        "train_max_target_year": int(train["next_year"].max()),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "selected_threshold": threshold,
    }

    model_metrics = {
        "model": "hist_gradient_boosting",
        **common,
        **complete_metrics(
            test[target_column],
            probability,
            prediction,
        ),
    }
    baseline_metrics = {
        "model": "always_no_event_baseline",
        **common,
        **complete_metrics(
            test[target_column],
            baseline_probability,
            baseline_prediction,
        ),
    }

    identity_columns = [
        column
        for column in [
            "person_key",
            "company",
            "GV_KEY",
            "full_name",
            "year",
            "next_year",
        ]
        if column in test.columns
    ]

    predictions = test[identity_columns].copy()
    predictions["marker"] = marker
    predictions["direction"] = direction
    predictions["target_event"] = test[target_column].to_numpy()
    predictions["predicted_probability"] = probability
    predictions["selected_threshold"] = threshold
    predictions["predicted_class"] = prediction
    predictions["baseline_probability"] = baseline_probability

    threshold_table.insert(0, "marker", marker)
    threshold_table.insert(1, "direction", direction)
    threshold_table.insert(2, "target_year", test_year)

    elapsed = perf_counter() - started

    progress(
        f"{marker} {direction}, target year {test_year}: READY — "
        f"PR-AUC={model_metrics['pr_auc']:.4f}, "
        f"baseline={model_metrics['positive_share']:.4f}, "
        f"ROC-AUC={model_metrics['roc_auc']:.4f}, "
        f"top-10% lift={model_metrics['top_10_lift']:.2f}, "
        f"{elapsed:.1f} sec."
    )

    return {
        "metrics": [model_metrics, baseline_metrics],
        "predictions": predictions,
        "thresholds": threshold_table,
    }


# ============================================================
# POOLED AND STABILITY SUMMARIES
# ============================================================

def pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate pooled out-of-sample metrics for every target."""

    rows = []

    for (marker, direction), part in predictions.groupby(
        ["marker", "direction"],
        sort=False,
    ):
        y_true = part["target_event"].to_numpy()
        probability = part["predicted_probability"].to_numpy()
        prediction = part["predicted_class"].to_numpy()

        rows.append({
            "marker": marker,
            "direction": direction,
            "model": "hist_gradient_boosting",
            **complete_metrics(y_true, probability, prediction),
        })

        constant_probability = np.full(
            len(part),
            float(np.mean(y_true)),
        )
        zero_prediction = np.zeros(len(part), dtype=int)

        rows.append({
            "marker": marker,
            "direction": direction,
            "model": "constant_prevalence_reference",
            **complete_metrics(
                y_true,
                constant_probability,
                zero_prediction,
            ),
        })

    return pd.DataFrame(rows)


def stability_summary(yearly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize variation in model quality across test years."""

    model = yearly_metrics[
        yearly_metrics["model"] == "hist_gradient_boosting"
    ]

    metric_columns = [
        "pr_auc",
        "pr_auc_over_prevalence",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "top_5_lift",
        "top_10_lift",
        "top_20_lift",
        "top_10_event_capture",
    ]

    rows = []

    for (marker, direction), part in model.groupby(
        ["marker", "direction"],
        sort=False,
    ):
        for metric in metric_columns:
            rows.append({
                "marker": marker,
                "direction": direction,
                "metric": metric,
                "mean_across_years": float(part[metric].mean()),
                "std_across_years": float(part[metric].std()),
                "minimum": float(part[metric].min()),
                "maximum": float(part[metric].max()),
            })

    return pd.DataFrame(rows)


def target_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table used for the console and article review."""

    model = pooled[
        pooled["model"] == "hist_gradient_boosting"
    ].copy()

    columns = [
        "marker",
        "direction",
        "n",
        "n_positive",
        "positive_share",
        "pr_auc",
        "pr_auc_over_prevalence",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "top_10_precision",
        "top_10_lift",
        "top_10_event_capture",
    ]

    return model[columns].sort_values(
        ["pr_auc_over_prevalence", "roc_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    total_started = perf_counter()

    progress("STEP 5.1/6: Loading and validating transition data...")
    data = load_data(INPUT_FILE)
    validate_data(data)

    progress("STEP 5.2/6: Building leakage-safe common features...")
    common = prepare_common_features(data)
    progress(
        f"Common features ready for {len(common):,} transitions."
    )

    all_metrics = []
    all_predictions = []
    all_thresholds = []
    target_populations = []

    progress(
        f"STEP 5.3/6: Running {len(TARGETS)} targets across "
        f"{len(TEST_YEARS)} rolling test years..."
    )

    for target_number, (marker, direction) in enumerate(
        TARGETS,
        start=1,
    ):
        progress(
            f"TARGET {target_number}/{len(TARGETS)}: "
            f"{marker} {direction}."
        )

        target_data, features, target_column = build_target_data(
            common,
            marker,
            direction,
        )

        target_populations.append({
            "marker": marker,
            "direction": direction,
            "n_eligible": len(target_data),
            "n_events": int(target_data[target_column].sum()),
            "event_rate": float(target_data[target_column].mean()),
            "n_features": len(features),
        })

        for fold_number, test_year in enumerate(
            TEST_YEARS,
            start=1,
        ):
            progress(
                f"Target {target_number}/{len(TARGETS)}, "
                f"fold {fold_number}/{len(TEST_YEARS)}."
            )

            result = run_fold(
                target_data,
                features,
                target_column,
                marker,
                direction,
                test_year,
            )

            all_metrics.extend(result["metrics"])
            all_predictions.append(result["predictions"])
            all_thresholds.append(result["thresholds"])

    progress("STEP 5.4/6: Combining out-of-sample results...")

    yearly_metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    thresholds = pd.concat(all_thresholds, ignore_index=True)
    populations = pd.DataFrame(target_populations)

    pooled = pooled_summary(predictions)
    stability = stability_summary(yearly_metrics)
    compact = target_summary(pooled)

    progress("STEP 5.5/6: Saving the result workbook...")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        compact.to_excel(
            writer,
            sheet_name="target_summary",
            index=False,
        )
        populations.to_excel(
            writer,
            sheet_name="target_populations",
            index=False,
        )
        yearly_metrics.to_excel(
            writer,
            sheet_name="yearly_metrics",
            index=False,
        )
        pooled.to_excel(
            writer,
            sheet_name="pooled_metrics",
            index=False,
        )
        stability.to_excel(
            writer,
            sheet_name="stability_summary",
            index=False,
        )
        thresholds.to_excel(
            writer,
            sheet_name="validation_thresholds",
            index=False,
        )
        predictions.to_excel(
            writer,
            sheet_name="test_predictions",
            index=False,
        )

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

    progress(
        f"Result saved to: {OUTPUT_FILE.resolve()}"
    )

    progress("STEP 5.6/6: Printing pooled target results...")

    print("\nPOOLED MARKER-SPECIFIC RESULTS", flush=True)
    print(
        compact.round(4).to_string(index=False),
        flush=True,
    )

    total_elapsed = perf_counter() - total_started
    progress(
        f"STEP 5 COMPLETE. Total time: "
        f"{total_elapsed / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
