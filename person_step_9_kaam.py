"""Step 9: temporal Kolmogorov-Arnold Additive Model (KAAM).

The model used here is the additive, interpretable form described in the
article:

    logit P(Y=1 | x) = beta_0 + sum_j f_j(x_j),

where every f_j is a trainable one-dimensional B-spline function.  This is
the additive KAN/KAAM branch: it can learn nonlinear effects while keeping
the contribution of every input feature separate.  No future observations
are used for feature construction, hyperparameter selection or threshold
selection.

Put this file next to:
  * person_step_7_gower_hybrid.py
  * person_step_2_transitions.xlsx
  * 20260721_OD_Database_1993_2024b.xlsx

Optional comparison input:
  * person_step_7_gower_hybrid_result.xlsx

Run the full experiment:
    python person_step_9_kaam.py

By default the full experiment uses every chronologically valid test year in
the database.  A year is valid when it has a previous-year validation fold,
at least three earlier training years, and both target classes in every fold.

Optional restriction to selected test years:
    $env:STEP9_YEARS = "2019,2020,2021,2022,2023"
    python person_step_9_kaam.py

Quick technical test (one target, test year 2023):
    $env:STEP9_SMOKE = "1"
    python person_step_9_kaam.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.class_weight import compute_sample_weight

import person_step_7_gower_hybrid as step7


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path("person_step_9_kaam_result.xlsx")
STEP7_RESULT = Path("person_step_7_gower_hybrid_result.xlsx")

RANDOM_STATE = 42
SMOKE_TEST = os.getenv("STEP9_SMOKE", "0") == "1"
YEAR_OVERRIDE = os.getenv("STEP9_YEARS", "").strip()
MIN_TRAIN_YEARS = 3
FULL_TARGETS = [
    ("evp", "onset"),
    ("support", "loss"),
    ("primary", "onset"),
    ("bu", "loss"),
    ("cxo", "loss"),
]
TARGETS = FULL_TARGETS[:1] if SMOKE_TEST else FULL_TARGETS

# The grid is deliberately compact.  Selection is performed only on the
# previous-year validation fold and PR-AUC is the primary criterion.
KAAM_CANDIDATES = [
    {"candidate_id": 1, "n_knots": 3, "degree": 2, "C": 0.10},
    {"candidate_id": 2, "n_knots": 4, "degree": 2, "C": 0.30},
    {"candidate_id": 3, "n_knots": 5, "degree": 3, "C": 0.30},
    {"candidate_id": 4, "n_knots": 6, "degree": 3, "C": 1.00},
]
if SMOKE_TEST:
    KAAM_CANDIDATES = KAAM_CANDIDATES[:2]

IMPORTANCE_SAMPLE_SIZE = 600 if SMOKE_TEST else 2000
IMPORTANCE_REPEATS = 1 if SMOKE_TEST else 2
BOOTSTRAP_REPEATS = 50 if SMOKE_TEST else 1000


def progress(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def resolve_test_years(population: pd.DataFrame) -> list[int]:
    """Return all leakage-safe rolling years available for one target."""
    available = sorted(
        pd.to_numeric(population["next_year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if SMOKE_TEST:
        candidates = [2023]
    elif YEAR_OVERRIDE:
        candidates = sorted({
            int(value.strip())
            for value in YEAR_OVERRIDE.split(",")
            if value.strip()
        })
    else:
        candidates = available

    valid = []
    for test_year in candidates:
        validation_year = test_year - 1
        train = population[population["next_year"] < validation_year]
        validation = population[
            population["next_year"] == validation_year
        ]
        test = population[population["next_year"] == test_year]
        if train["next_year"].nunique() < MIN_TRAIN_YEARS:
            continue
        if any(part.empty for part in [train, validation, test]):
            continue
        if any(
            part["target_event"].nunique() < 2
            for part in [train, validation, test]
        ):
            continue
        valid.append(test_year)
    return valid


def feature_types(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Separate continuous, binary and constant predictors using train only."""
    continuous, binary, constant = [], [], []
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce")
        unique = values.dropna().nunique()
        if unique <= 1:
            constant.append(feature)
        elif unique == 2:
            binary.append(feature)
        else:
            continuous.append(feature)
    return continuous, binary, constant


def build_kaam(
    train: pd.DataFrame,
    features: list[str],
    parameters: dict,
) -> tuple[Pipeline, list[str], list[str], list[str]]:
    """Create a sparse additive spline-logistic model."""
    continuous, binary, constant = feature_types(train, features)
    transformers = []

    if continuous:
        continuous_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "spline",
                SplineTransformer(
                    n_knots=int(parameters["n_knots"]),
                    degree=int(parameters["degree"]),
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                    sparse_output=True,
                ),
            ),
        ])
        transformers.append(("continuous", continuous_pipeline, continuous))

    if binary:
        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("binary", binary_pipeline, binary))

    if not transformers:
        raise ValueError("KAAM has no non-constant predictors.")

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    classifier = LogisticRegression(
        C=float(parameters["C"]),
        l1_ratio=0.0,
        solver="liblinear",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model = Pipeline([
        ("features", preprocessing),
        ("classifier", classifier),
    ])
    return model, continuous, binary, constant


def fit_kaam(
    train: pd.DataFrame,
    features: list[str],
    parameters: dict,
) -> tuple[Pipeline, list[str], list[str], list[str]]:
    model, continuous, binary, constant = build_kaam(
        train, features, parameters
    )
    weights = compute_sample_weight(
        class_weight="balanced",
        y=train["target_event"],
    )
    model.fit(
        train[features],
        train["target_event"],
        classifier__sample_weight=weights,
    )
    return model, continuous, binary, constant


def select_kaam(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    marker: str,
    direction: str,
    test_year: int,
) -> tuple[dict, float, pd.DataFrame]:
    rows = []
    probabilities = {}
    for number, parameters in enumerate(KAAM_CANDIDATES, start=1):
        started = perf_counter()
        progress(
            f"    KAAM candidate {number}/{len(KAAM_CANDIDATES)}: "
            f"knots={parameters['n_knots']}, degree={parameters['degree']}, "
            f"C={parameters['C']}..."
        )
        model, continuous, binary, constant = fit_kaam(
            train, features, parameters
        )
        probability = model.predict_proba(validation[features])[:, 1]
        threshold = step7.select_threshold(
            validation["target_event"], probability
        )
        metrics = step7.complete_metrics(
            validation["target_event"], probability, threshold
        )
        candidate_id = int(parameters["candidate_id"])
        probabilities[candidate_id] = probability
        rows.append({
            "marker": marker,
            "direction": direction,
            "test_year": test_year,
            **parameters,
            "n_continuous_features": len(continuous),
            "n_binary_features": len(binary),
            "n_constant_features": len(constant),
            "selected_threshold": threshold,
            "validation_pr_auc": metrics["pr_auc"],
            "validation_pr_auc_over_prevalence": metrics[
                "pr_auc_over_prevalence"
            ],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_f1": metrics["f1"],
            "elapsed_seconds": perf_counter() - started,
        })

    table = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc", "validation_f1"],
        ascending=[False, False, False],
    )
    best = table.iloc[0]
    best_id = int(best["candidate_id"])
    best_parameters = next(
        candidate for candidate in KAAM_CANDIDATES
        if int(candidate["candidate_id"]) == best_id
    )
    return best_parameters, float(best["selected_threshold"]), table


def effect_curves(
    model: Pipeline,
    development: pd.DataFrame,
    features: list[str],
    continuous: list[str],
    binary: list[str],
    marker: str,
    direction: str,
    test_year: int,
) -> pd.DataFrame:
    """Extract model-implied one-feature-at-a-time additive effects."""
    baseline_values = {}
    for feature in features:
        values = pd.to_numeric(development[feature], errors="coerce")
        if feature in binary:
            mode = values.mode(dropna=True)
            baseline_values[feature] = float(mode.iloc[0]) if len(mode) else 0.0
        else:
            baseline_values[feature] = float(values.median())

    baseline = pd.DataFrame(
        [baseline_values], columns=features, dtype=float
    )
    baseline_logit = float(model.decision_function(baseline)[0])
    rows = []

    for feature in features:
        values = pd.to_numeric(
            development[feature], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        if feature in binary:
            grid = np.sort(values.unique())
        elif feature in continuous:
            grid = np.unique(
                values.quantile(np.linspace(0.0, 1.0, 11)).to_numpy()
            )
        else:
            continue

        scenarios = pd.concat(
            [baseline] * len(grid), ignore_index=True
        )
        scenarios[feature] = grid
        logits = model.decision_function(scenarios)
        probabilities = model.predict_proba(scenarios)[:, 1]
        for value, logit, probability in zip(grid, logits, probabilities):
            rows.append({
                "marker": marker,
                "direction": direction,
                "test_year": test_year,
                "feature": feature,
                "feature_type": (
                    "binary" if feature in binary else "continuous"
                ),
                "feature_value": float(value),
                "log_odds_effect_vs_median": float(logit - baseline_logit),
                "scenario_probability": float(probability),
                "baseline_probability": float(
                    model.predict_proba(baseline)[0, 1]
                ),
            })
    return pd.DataFrame(rows)


def run_fold(
    population: pd.DataFrame,
    features: list[str],
    marker: str,
    direction: str,
    test_year: int,
) -> dict:
    progress(f"  {marker} {direction}, test {test_year}: splitting...")
    train, validation, test = step7.split_fold(population, test_year)
    progress(
        f"    train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}."
    )

    parameters, threshold, tuning = select_kaam(
        train, validation, features, marker, direction, test_year
    )
    development = pd.concat([train, validation], ignore_index=True)
    progress("    refitting selected KAAM on train + validation...")
    model, continuous, binary, constant = fit_kaam(
        development, features, parameters
    )
    probability = model.predict_proba(test[features])[:, 1]
    metrics = {
        "marker": marker,
        "direction": direction,
        "test_year": test_year,
        "model": "kaam_additive_splines",
        **parameters,
        "selected_threshold": threshold,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "n_continuous_features": len(continuous),
        "n_binary_features": len(binary),
        "n_constant_features": len(constant),
        **step7.complete_metrics(
            test["target_event"], probability, threshold
        ),
    }

    predictions = test[[
        "person_key", "company", "GV_KEY", "full_name", "role",
        "year", "next_year", "target_event",
    ]].copy()
    predictions["marker"] = marker
    predictions["direction"] = direction
    predictions["test_year"] = test_year
    predictions["probability_kaam"] = probability
    predictions["threshold_kaam"] = threshold
    predictions["class_kaam"] = (probability >= threshold).astype("int8")

    progress("    calculating permutation importance...")
    importance_sample = test
    if len(test) > IMPORTANCE_SAMPLE_SIZE:
        importance_sample = test.sample(
            IMPORTANCE_SAMPLE_SIZE, random_state=RANDOM_STATE
        )
    importance_result = permutation_importance(
        model,
        importance_sample[features],
        importance_sample["target_event"],
        scoring="average_precision",
        n_repeats=IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "marker": marker,
        "direction": direction,
        "test_year": test_year,
        "feature": features,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    curves = effect_curves(
        model, development, features, continuous, binary,
        marker, direction, test_year,
    )
    progress(
        f"    READY: PR-AUC={metrics['pr_auc']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"top-10% lift={metrics['top_10_lift']:.2f}."
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "tuning": tuning,
        "importance": importance,
        "curves": curves,
    }


def pooled_kaam_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (marker, direction), group in predictions.groupby(
        ["marker", "direction"], sort=False
    ):
        rows.append({
            "marker": marker,
            "direction": direction,
            "model": "kaam_additive_splines",
            **step7.complete_metrics(
                group["target_event"],
                group["probability_kaam"],
                prediction=group["class_kaam"],
            ),
        })
    return pd.DataFrame(rows)


def compare_with_step7(
    kaam_pooled: pd.DataFrame,
    kaam_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not STEP7_RESULT.exists():
        progress(
            "Step 7 result not found; model comparison and paired bootstrap "
            "will be skipped."
        )
        return kaam_pooled.copy(), pd.DataFrame()

    old_pooled = pd.read_excel(STEP7_RESULT, sheet_name="pooled_metrics")
    old_predictions = pd.read_excel(
        STEP7_RESULT, sheet_name="test_predictions"
    )
    keys = ["person_key", "marker", "direction", "next_year"]
    old_keep = [
        *keys, "GV_KEY", "target_event", "probability_base_boosting",
    ]
    merged = kaam_predictions.merge(
        old_predictions[old_keep],
        on=keys,
        how="inner",
        suffixes=("", "_step7"),
        validate="one_to_one",
    )
    if merged.empty:
        progress("No matching Step 7 predictions; paired bootstrap skipped.")
        return kaam_pooled.copy(), pd.DataFrame()

    # Step 7 normally covers 2019-2023, whereas the full Step 9 run covers
    # every valid year.  Recalculate KAAM metrics on the overlapping rows so
    # that the comparison sheet never mixes different evaluation periods.
    overlap_rows = []
    for (marker, direction), group in merged.groupby(
        ["marker", "direction"], sort=False
    ):
        overlap_rows.append({
            "marker": marker,
            "direction": direction,
            "model": "kaam_additive_splines",
            **step7.complete_metrics(
                group["target_event"],
                group["probability_kaam"],
                prediction=group["class_kaam"],
            ),
        })
    comparison = pd.concat(
        [old_pooled, pd.DataFrame(overlap_rows)],
        ignore_index=True,
        sort=False,
    )

    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for (marker, direction), group in merged.groupby(
        ["marker", "direction"], sort=False
    ):
        # Use local positional indices.  The original merged index would be
        # out of bounds for the second and subsequent target groups.
        group = group.reset_index(drop=True)
        y = group["target_event"].to_numpy(dtype=int)
        p_kaam = group["probability_kaam"].to_numpy(dtype=float)
        p_base = group["probability_base_boosting"].to_numpy(dtype=float)
        point_pr = average_precision_score(y, p_kaam) - average_precision_score(
            y, p_base
        )
        point_roc = roc_auc_score(y, p_kaam) - roc_auc_score(y, p_base)

        company_groups = {
            key: indices.to_numpy()
            for key, indices in group.groupby("GV_KEY").groups.items()
        }
        companies = np.array(list(company_groups), dtype=object)
        pr_differences, roc_differences = [], []
        for _ in range(BOOTSTRAP_REPEATS):
            sampled = rng.choice(companies, size=len(companies), replace=True)
            indices = np.concatenate([company_groups[key] for key in sampled])
            sample_y = y[indices]
            if np.unique(sample_y).size < 2:
                continue
            pr_differences.append(
                average_precision_score(sample_y, p_kaam[indices])
                - average_precision_score(sample_y, p_base[indices])
            )
            roc_differences.append(
                roc_auc_score(sample_y, p_kaam[indices])
                - roc_auc_score(sample_y, p_base[indices])
            )
        rows.append({
            "marker": marker,
            "direction": direction,
            "comparison": "KAAM minus base_boosting",
            "n": len(group),
            "n_companies": len(companies),
            "pr_auc_difference": point_pr,
            "pr_auc_ci_low": np.percentile(pr_differences, 2.5),
            "pr_auc_ci_high": np.percentile(pr_differences, 97.5),
            "pr_auc_probability_positive": np.mean(
                np.asarray(pr_differences) > 0
            ),
            "roc_auc_difference": point_roc,
            "roc_auc_ci_low": np.percentile(roc_differences, 2.5),
            "roc_auc_ci_high": np.percentile(roc_differences, 97.5),
            "roc_auc_probability_positive": np.mean(
                np.asarray(roc_differences) > 0
            ),
            "bootstrap_repeats": len(pr_differences),
        })
    return comparison, pd.DataFrame(rows)


def importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (marker, direction, feature), group in importance.groupby(
        ["marker", "direction", "feature"], sort=False
    ):
        values = group["importance_mean"]
        rows.append({
            "marker": marker,
            "direction": direction,
            "feature": feature,
            "mean_importance": values.mean(),
            "std_across_years": values.std(),
            "minimum": values.min(),
            "maximum": values.max(),
            "positive_years": int((values > 0).sum()),
            "n_years": len(values),
            "positive_year_share": float((values > 0).mean()),
        })
    summary = pd.DataFrame(rows)
    summary["rank_within_target"] = summary.groupby(
        ["marker", "direction"]
    )["mean_importance"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(
        ["marker", "direction", "rank_within_target"]
    )


def save_results(
    yearly: pd.DataFrame,
    pooled: pd.DataFrame,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    tuning: pd.DataFrame,
    importance: pd.DataFrame,
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    readme = pd.DataFrame({
        "item": [
            "Model",
            "Equation",
            "Temporal design",
            "Hyperparameter selection",
            "Primary metric",
            "Interpretation",
            "Causal limitation",
        ],
        "definition": [
            "Kolmogorov-Arnold Additive Model implemented as separate one-dimensional B-spline functions plus logistic output.",
            "logit P(Y=1|x) = beta_0 + sum_j f_j(x_j).",
            "Every valid database year is tested using an expanding training window, previous-year validation and next-year test; at least three earlier training years are required.",
            "Number of knots, spline degree and L2 strength are selected on validation PR-AUC only.",
            "PR-AUC and PR-AUC/prevalence; ROC-AUC and top-decile lift are secondary.",
            "Permutation importance and learned one-feature effect curves.",
            "Predictive effects do not establish causal effects of organizational factors.",
        ],
    })
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="readme", index=False)
        pooled.to_excel(writer, sheet_name="kaam_pooled", index=False)
        yearly.to_excel(writer, sheet_name="kaam_yearly", index=False)
        comparison.to_excel(writer, sheet_name="comparison_step7", index=False)
        paired.to_excel(writer, sheet_name="paired_bootstrap", index=False)
        tuning.to_excel(writer, sheet_name="validation_tuning", index=False)
        importance.to_excel(
            writer, sheet_name="permutation_importance", index=False
        )
        importance_summary(importance).to_excel(
            writer, sheet_name="importance_summary", index=False
        )
        curves.to_excel(writer, sheet_name="effect_curves", index=False)
        predictions.to_excel(
            writer, sheet_name="test_predictions", index=False
        )


def main() -> None:
    started = perf_counter()
    progress("STEP 9.1/6: loading data and Step 7 feature functions...")
    transitions, raw = step7.load_inputs()
    raw_features, raw_company_columns = step7.build_raw_features(raw)
    # Step 7 versions created before this script may contain the same helper
    # column twice.  Deduplicate defensively without changing its values.
    raw_features = raw_features.loc[:, ~raw_features.columns.duplicated()].copy()
    raw_company_columns = list(dict.fromkeys(raw_company_columns))
    common = step7.prepare_common_features(
        transitions, raw_features, raw_company_columns
    )

    metrics, predictions, tuning = [], [], []
    importance, curves = [], []
    progress("STEP 9.2/6: running rolling KAAM experiments...")
    for target_number, (marker, direction) in enumerate(TARGETS, start=1):
        progress(
            f"TARGET {target_number}/{len(TARGETS)}: {marker} {direction}."
        )
        population, _, enhanced_features = step7.build_target_data(
            common, marker, direction, raw_company_columns
        )
        test_years = resolve_test_years(population)
        if not test_years:
            raise ValueError(
                f"No valid rolling years for {marker} {direction}."
            )
        progress(
            "  valid test years: "
            + ", ".join(map(str, test_years))
            + "."
        )
        for fold_number, test_year in enumerate(test_years, start=1):
            progress(
                f"  fold {fold_number}/{len(test_years)}: {test_year}."
            )
            result = run_fold(
                population, enhanced_features, marker, direction, test_year
            )
            metrics.append(result["metrics"])
            predictions.append(result["predictions"])
            tuning.append(result["tuning"])
            importance.append(result["importance"])
            curves.append(result["curves"])

    progress("STEP 9.3/6: pooling out-of-sample predictions...")
    yearly = pd.DataFrame(metrics)
    all_predictions = pd.concat(predictions, ignore_index=True)
    pooled = pooled_kaam_metrics(all_predictions)

    progress("STEP 9.4/6: comparing KAAM with Step 7...")
    comparison, paired = compare_with_step7(pooled, all_predictions)

    progress("STEP 9.5/6: saving tables and interpretations...")
    save_results(
        yearly=yearly,
        pooled=pooled,
        comparison=comparison,
        paired=paired,
        tuning=pd.concat(tuning, ignore_index=True),
        importance=pd.concat(importance, ignore_index=True),
        curves=pd.concat(curves, ignore_index=True),
        predictions=all_predictions,
    )

    progress("STEP 9.6/6: complete.")
    print("\nKAAM POOLED RESULTS", flush=True)
    print(
        pooled[[
            "marker", "direction", "n", "positive_share", "pr_auc",
            "pr_auc_over_prevalence", "roc_auc", "f1", "precision",
            "recall", "top_10_lift", "top_10_event_capture",
        ]].round(4).to_string(index=False),
        flush=True,
    )
    if not paired.empty:
        print("\nPAIRED COMPARISON WITH BASE BOOSTING", flush=True)
        print(
            paired[[
                "marker", "direction", "pr_auc_difference",
                "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc_difference",
                "roc_auc_ci_low", "roc_auc_ci_high",
            ]].round(4).to_string(index=False),
            flush=True,
        )
    progress(
        f"Saved to {OUTPUT_FILE.resolve()}. "
        f"Total time: {(perf_counter() - started) / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
