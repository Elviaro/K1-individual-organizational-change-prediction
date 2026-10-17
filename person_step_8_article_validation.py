"""
STEP 8. Article-grade validation of the individual marker models.

The script uses strictly out-of-sample predictions saved by Step 7 and adds:

1. Prequential (previous-year) model selection without looking at the current
   test year.
2. Company-cluster bootstrap confidence intervals for PR-AUC and ROC-AUC.
3. Paired bootstrap comparisons against the base boosting model.
4. Year-to-year stability tables.
5. Aggregated permutation importance across rolling folds.
6. A genuine progressive feature ablation:
      position only -> + individual history -> + company context
      -> + target-state and lagged dynamics.

Expected input files in the project directory:
    person_step_7_gower_hybrid_result.xlsx
    person_step_7_gower_hybrid.py
    person_step_2_transitions.xlsx
    20260721_OD_Database_1993_2024b.xlsx

Output:
    person_step_8_article_validation_result.xlsx

Environment switches:
    STEP8_BOOTSTRAPS=1000       number of cluster bootstrap repetitions
    STEP8_SKIP_ABLATION=1      skip model refitting and analyse Step 7 only
    STEP8_SMOKE=1              50 bootstraps, first target, 2023 ablation only
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
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


RANDOM_STATE = 42
OUTPUT_FILE = Path(os.getenv(
    "STEP8_OUTPUT_FILE", "person_step_8_article_validation_result.xlsx"
))
STEP7_RESULT = Path(os.getenv(
    "STEP8_STEP7_RESULT", "person_step_7_gower_hybrid_result.xlsx"
))
TRANSITIONS_FILE = Path(os.getenv(
    "STEP8_TRANSITIONS_FILE", "person_step_2_transitions.xlsx"
))
RAW_FILE = Path(os.getenv(
    "STEP8_RAW_FILE", "20260721_OD_Database_1993_2024b.xlsx"
))

MODEL_NAMES = [
    "base_boosting",
    "enhanced_boosting",
    "gower_precedents",
    "role_text",
    "hybrid",
]
SELECTION_METRIC = "pr_auc"
SMOKE_TEST = os.getenv("STEP8_SMOKE", "0") == "1"
SKIP_ABLATION = os.getenv("STEP8_SKIP_ABLATION", "0") == "1"
N_BOOTSTRAPS = int(os.getenv("STEP8_BOOTSTRAPS", "50" if SMOKE_TEST else "1000"))
MIN_VALID_BOOTSTRAPS = max(20, int(0.8 * N_BOOTSTRAPS))


def progress(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def resolve_file(path: Path) -> Path:
    """Resolve both the normal PyCharm layout and this review workspace."""
    candidates = [
        path,
        Path("upload") / path.name,
        Path("project_sources") / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Required file not found: {path.name}. Put it next to this script."
    )


def safe_roc_auc(y_true, probability) -> float:
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, probability))


def top_fraction_metrics(y_true, probability, share: float = 0.10) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    n_selected = max(1, int(np.ceil(len(y) * share)))
    selected = np.argsort(-probability, kind="stable")[:n_selected]
    prevalence = float(y.mean())
    precision = float(y[selected].mean())
    return {
        "top_10_precision": precision,
        "top_10_lift": precision / prevalence if prevalence else np.nan,
        "top_10_event_capture": (
            float(y[selected].sum() / y.sum()) if y.sum() else np.nan
        ),
    }


def select_threshold(y_true, probability) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probability
    )
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # F1 first; precision, recall and the higher threshold resolve ties.
    best = np.lexsort((thresholds, recall, precision, f1))[-1]
    return float(thresholds[best])


def complete_metrics(
    y_true,
    probability,
    prediction=None,
    threshold: float | None = None,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if prediction is None:
        if threshold is None:
            raise ValueError("Provide prediction or threshold.")
        prediction = (probability >= threshold).astype(int)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, probability))
    result = {
        "n": len(y),
        "n_positive": int(y.sum()),
        "positive_share": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_over_prevalence": (
            pr_auc / prevalence if prevalence else np.nan
        ),
        "roc_auc": safe_roc_auc(y, probability),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result.update(top_fraction_metrics(y, probability))
    return result


def load_step7_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = resolve_file(STEP7_RESULT)
    progress(f"STEP 8.1/7: loading {path.name}...")
    predictions = pd.read_excel(path, sheet_name="test_predictions")
    importance = pd.read_excel(path, sheet_name="permutation_importance")

    required = {
        "company", "GV_KEY", "year", "next_year", "target_event",
        "marker", "direction",
    }
    for model in MODEL_NAMES:
        required.update({
            f"probability_{model}",
            f"class_{model}",
        })
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            "Step 7 test_predictions is missing: " + ", ".join(missing)
        )
    predictions["target_event"] = predictions["target_event"].astype(int)
    predictions["next_year"] = predictions["next_year"].astype(int)
    progress(
        f"Loaded {len(predictions):,} out-of-sample predictions for "
        f"{predictions[['marker', 'direction']].drop_duplicates().shape[0]} targets."
    )
    return predictions, importance


def score_model(group: pd.DataFrame, model: str) -> tuple[float, float]:
    y = group["target_event"].to_numpy()
    probability = group[f"probability_{model}"].to_numpy()
    return (
        float(average_precision_score(y, probability)),
        safe_roc_auc(y, probability),
    )


def build_prequential_selection(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select the branch for year t only from year t-1 performance.

    The earliest available year uses the prespecified base branch.  The class
    threshold remains the Step 7 threshold selected on that fold's validation
    year, so neither branch nor threshold sees the current test labels.
    """
    progress("STEP 8.2/7: applying previous-year model selection...")
    selected_parts = []
    selection_rows = []
    yearly_rows = []

    keys = ["marker", "direction"]
    for (marker, direction), target in predictions.groupby(keys, sort=True):
        target = target.sort_values("next_year").copy()
        years = sorted(target["next_year"].unique())
        if SMOKE_TEST:
            years = years[-1:]
        all_target = predictions[
            (predictions["marker"] == marker)
            & (predictions["direction"] == direction)
        ]

        for position, test_year in enumerate(years):
            current = target[target["next_year"] == test_year].copy()
            previous_year = test_year - 1
            prior = all_target[all_target["next_year"] == previous_year]

            candidate_rows = []
            if prior.empty:
                selected_model = "base_boosting"
                selection_basis = "prespecified_first_year"
            else:
                for model in MODEL_NAMES:
                    pr_auc, roc_auc = score_model(prior, model)
                    candidate_rows.append({
                        "marker": marker,
                        "direction": direction,
                        "test_year": test_year,
                        "selection_year": previous_year,
                        "candidate_model": model,
                        "selection_pr_auc": pr_auc,
                        "selection_roc_auc": roc_auc,
                    })
                ranked = sorted(
                    candidate_rows,
                    key=lambda row: (
                        row["selection_pr_auc"],
                        row["selection_roc_auc"],
                        -MODEL_NAMES.index(row["candidate_model"]),
                    ),
                    reverse=True,
                )
                selected_model = ranked[0]["candidate_model"]
                selection_basis = "previous_test_year"

            selection_rows.extend(candidate_rows)
            probability = current[
                f"probability_{selected_model}"
            ].to_numpy()
            prediction = current[f"class_{selected_model}"].to_numpy()
            current["selected_model"] = selected_model
            current["selected_probability"] = probability
            current["selected_class"] = prediction
            current["selection_year"] = (
                previous_year if not prior.empty else np.nan
            )
            current["selection_basis"] = selection_basis
            selected_parts.append(current)

            yearly_rows.append({
                "marker": marker,
                "direction": direction,
                "test_year": test_year,
                "selected_model": selected_model,
                "selection_year": (
                    previous_year if not prior.empty else np.nan
                ),
                "selection_basis": selection_basis,
                **complete_metrics(
                    current["target_event"],
                    probability,
                    prediction=prediction,
                ),
            })

    selected = pd.concat(selected_parts, ignore_index=True)
    yearly = pd.DataFrame(yearly_rows)
    selection = pd.DataFrame(selection_rows)

    pooled_rows = []
    for (marker, direction), group in selected.groupby(keys, sort=True):
        pooled_rows.append({
            "marker": marker,
            "direction": direction,
            "strategy": "prequential_selected",
            **complete_metrics(
                group["target_event"],
                group["selected_probability"],
                prediction=group["selected_class"],
            ),
        })
    pooled = pd.DataFrame(pooled_rows)
    return selected, selection, pd.concat(
        [yearly, pooled.rename(columns={"strategy": "selected_model"})],
        ignore_index=True,
        sort=False,
    )


def _cluster_index_groups(group: pd.DataFrame) -> list[np.ndarray]:
    cluster_column = "GV_KEY" if "GV_KEY" in group.columns else "company"
    return [
        np.asarray(indices, dtype=int)
        for indices in group.groupby(cluster_column, sort=False).indices.values()
    ]


def _bootstrap_indices(
    clusters: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[index] for index in selected])


def bootstrap_confidence_intervals(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress(
        f"STEP 8.3/7: running {N_BOOTSTRAPS:,} company-cluster bootstraps..."
    )
    ci_rows = []
    comparison_rows = []
    keys = ["marker", "direction"]

    for target_number, ((marker, direction), group) in enumerate(
        selected.groupby(keys, sort=True), start=1
    ):
        group = group.reset_index(drop=True)
        clusters = _cluster_index_groups(group)
        y = group["target_event"].to_numpy(dtype=int)
        probability_map = {
            model: group[f"probability_{model}"].to_numpy(dtype=float)
            for model in MODEL_NAMES
        }
        probability_map["prequential_selected"] = group[
            "selected_probability"
        ].to_numpy(dtype=float)

        point = {
            model: {
                "pr_auc": float(average_precision_score(y, probability)),
                "roc_auc": safe_roc_auc(y, probability),
            }
            for model, probability in probability_map.items()
        }
        bootstrap_values = {
            model: {"pr_auc": [], "roc_auc": []}
            for model in probability_map
        }
        rng = np.random.default_rng(RANDOM_STATE + 1000 * target_number)

        for _ in range(N_BOOTSTRAPS):
            indices = _bootstrap_indices(clusters, rng)
            sampled_y = y[indices]
            if np.unique(sampled_y).size < 2:
                continue
            for model, probability in probability_map.items():
                sampled_probability = probability[indices]
                bootstrap_values[model]["pr_auc"].append(
                    average_precision_score(sampled_y, sampled_probability)
                )
                bootstrap_values[model]["roc_auc"].append(
                    roc_auc_score(sampled_y, sampled_probability)
                )

        valid_count = min(
            len(values["pr_auc"])
            for values in bootstrap_values.values()
        )
        if valid_count < MIN_VALID_BOOTSTRAPS:
            raise RuntimeError(
                f"Only {valid_count} valid bootstrap samples for "
                f"{marker} {direction}."
            )

        for model, metrics in bootstrap_values.items():
            for metric, values in metrics.items():
                array = np.asarray(values, dtype=float)
                ci_rows.append({
                    "marker": marker,
                    "direction": direction,
                    "model": model,
                    "metric": metric,
                    "estimate": point[model][metric],
                    "ci_95_low": float(np.quantile(array, 0.025)),
                    "ci_95_high": float(np.quantile(array, 0.975)),
                    "bootstrap_valid": len(array),
                    "cluster_unit": "company",
                })

        reference = "base_boosting"
        for model in [*MODEL_NAMES[1:], "prequential_selected"]:
            for metric in ["pr_auc", "roc_auc"]:
                candidate = np.asarray(
                    bootstrap_values[model][metric], dtype=float
                )
                baseline = np.asarray(
                    bootstrap_values[reference][metric], dtype=float
                )
                difference = candidate - baseline
                p_lower = (np.sum(difference <= 0) + 1) / (
                    len(difference) + 1
                )
                p_upper = (np.sum(difference >= 0) + 1) / (
                    len(difference) + 1
                )
                comparison_rows.append({
                    "marker": marker,
                    "direction": direction,
                    "candidate_model": model,
                    "reference_model": reference,
                    "metric": metric,
                    "candidate_estimate": point[model][metric],
                    "reference_estimate": point[reference][metric],
                    "difference": (
                        point[model][metric] - point[reference][metric]
                    ),
                    "difference_ci_95_low": float(
                        np.quantile(difference, 0.025)
                    ),
                    "difference_ci_95_high": float(
                        np.quantile(difference, 0.975)
                    ),
                    "two_sided_bootstrap_p": float(
                        min(1.0, 2 * min(p_lower, p_upper))
                    ),
                    "bootstrap_valid": len(difference),
                })
        progress(
            f"  Bootstrap {marker} {direction}: {valid_count:,} valid samples."
        )

    return pd.DataFrame(ci_rows), pd.DataFrame(comparison_rows)


def build_stability_tables(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress("STEP 8.4/7: summarising year-to-year stability...")
    yearly_rows = []
    keys = ["marker", "direction", "next_year"]
    for key, group in selected.groupby(keys, sort=True):
        marker, direction, year = key
        for model in [*MODEL_NAMES, "prequential_selected"]:
            if model == "prequential_selected":
                probability = group["selected_probability"]
                prediction = group["selected_class"]
            else:
                probability = group[f"probability_{model}"]
                prediction = group[f"class_{model}"]
            yearly_rows.append({
                "marker": marker,
                "direction": direction,
                "test_year": year,
                "model": model,
                **complete_metrics(
                    group["target_event"],
                    probability,
                    prediction=prediction,
                ),
            })
    yearly = pd.DataFrame(yearly_rows)

    summary_rows = []
    for (marker, direction, model), group in yearly.groupby(
        ["marker", "direction", "model"], sort=True
    ):
        base = yearly[
            (yearly["marker"] == marker)
            & (yearly["direction"] == direction)
            & (yearly["model"] == "base_boosting")
        ].set_index("test_year")
        aligned = group.set_index("test_year")
        row = {
            "marker": marker,
            "direction": direction,
            "model": model,
            "n_years": len(group),
        }
        for metric in [
            "pr_auc", "roc_auc", "f1", "precision", "recall",
            "balanced_accuracy", "top_10_lift",
        ]:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
            row[f"min_{metric}"] = float(group[metric].min())
            row[f"max_{metric}"] = float(group[metric].max())
        common_years = aligned.index.intersection(base.index)
        row["years_pr_auc_above_base"] = int(
            (
                aligned.loc[common_years, "pr_auc"]
                > base.loc[common_years, "pr_auc"]
            ).sum()
        )
        row["years_roc_auc_above_base"] = int(
            (
                aligned.loc[common_years, "roc_auc"]
                > base.loc[common_years, "roc_auc"]
            ).sum()
        )
        summary_rows.append(row)
    return yearly, pd.DataFrame(summary_rows)


def aggregate_importance(importance: pd.DataFrame) -> pd.DataFrame:
    progress("STEP 8.5/7: aggregating permutation importance...")
    if importance.empty:
        return importance
    required = {
        "marker", "direction", "target_year", "feature",
        "importance_mean", "importance_std",
    }
    missing = sorted(required.difference(importance.columns))
    if missing:
        raise ValueError(
            "Step 7 permutation_importance is missing: "
            + ", ".join(missing)
        )
    rows = []
    for (marker, direction, feature), group in importance.groupby(
        ["marker", "direction", "feature"], sort=True
    ):
        values = group["importance_mean"].to_numpy(dtype=float)
        rows.append({
            "marker": marker,
            "direction": direction,
            "feature": feature,
            "mean_importance": float(np.mean(values)),
            "median_importance": float(np.median(values)),
            "std_across_years": float(np.std(values, ddof=1))
            if len(values) > 1 else np.nan,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "positive_years": int(np.sum(values > 0)),
            "n_years": len(values),
            "positive_year_share": float(np.mean(values > 0)),
        })
    result = pd.DataFrame(rows)
    result["rank_within_target"] = result.groupby(
        ["marker", "direction"]
    )["mean_importance"].rank(method="dense", ascending=False)
    return result.sort_values(
        ["marker", "direction", "rank_within_target", "feature"]
    )


def _ablation_feature_sets(
    marker: str,
    base_features: list[str],
    enhanced_features: list[str],
    raw_company_columns: list[str],
) -> dict[str, list[str]]:
    """Return nested, substantively interpretable feature blocks."""
    position = [
        feature for feature in base_features
        if feature in {
            *[item for item in step7.MARKERS if item != marker],
            "current_marker_count",
            "multiple_current_markers",
        }
    ]
    individual_history_candidates = {
        "years_since_first_observation",
        "prior_transition_count",
        "prior_any_change_count",
        "prior_any_change_rate",
        "previous_transition_changed",
        "previous_transition_is_adjacent",
        "prior_target_event_count",
        "prior_target_event_rate",
        "previous_target_event",
    }
    company_context_candidates = {
        "company_prior_target_eligible",
        "company_prior_target_events",
        "company_prior_target_rate",
        *raw_company_columns,
    }
    dynamics_candidates = {
        "target_state_tenure",
        "prior_changed_marker_count",
        "prior_mean_markers_changed",
        "previous_transition_change_count",
        "lag_company_eligible",
        "lag_company_any_changes",
        "lag_company_marker_changes",
        "lag_company_any_change_rate",
        "lag_company_mean_marker_changes",
    }

    individual = [
        feature for feature in enhanced_features
        if feature in individual_history_candidates
    ]
    company = [
        feature for feature in enhanced_features
        if feature in company_context_candidates
    ]
    dynamics = [
        feature for feature in enhanced_features
        if feature in dynamics_candidates
    ]
    return {
        "A_position_only": list(dict.fromkeys(position)),
        "B_plus_individual_history": list(
            dict.fromkeys(position + individual)
        ),
        "C_plus_company_context": list(
            dict.fromkeys(position + individual + company)
        ),
        "D_plus_state_and_lagged_dynamics": list(
            dict.fromkeys(position + individual + company + dynamics)
        ),
    }


def run_ablation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    progress("STEP 8.6/7: running progressive feature ablation...")
    transitions_path = resolve_file(TRANSITIONS_FILE)
    raw_path = resolve_file(RAW_FILE)
    step7.TRANSITIONS_FILE = transitions_path
    step7.RAW_FILE = raw_path

    transitions, raw = step7.load_inputs()
    raw_features, raw_company_columns = step7.build_raw_features(raw)
    common = step7.prepare_common_features(
        transitions, raw_features, raw_company_columns
    )

    targets = step7.TARGETS[:1] if SMOKE_TEST else step7.TARGETS
    years = [2023] if SMOKE_TEST else step7.TEST_YEARS
    metric_rows = []
    prediction_rows = []
    feature_rows = []

    for target_number, (marker, direction) in enumerate(targets, start=1):
        progress(
            f"  Ablation target {target_number}/{len(targets)}: "
            f"{marker} {direction}."
        )
        population, base_features, enhanced_features = step7.build_target_data(
            common, marker, direction, raw_company_columns
        )
        feature_sets = _ablation_feature_sets(
            marker, base_features, enhanced_features, raw_company_columns
        )
        for block, features in feature_sets.items():
            for feature in features:
                feature_rows.append({
                    "marker": marker,
                    "direction": direction,
                    "feature_block": block,
                    "feature": feature,
                })

        for fold_number, test_year in enumerate(years, start=1):
            progress(
                f"    {marker} {direction}: year {test_year} "
                f"({fold_number}/{len(years)})."
            )
            train, validation, test = step7.split_fold(
                population, test_year
            )
            development = pd.concat(
                [train, validation], ignore_index=True
            )
            for block, features in feature_sets.items():
                validation_model = step7.fit_boosting(train, features)
                validation_probability = validation_model.predict_proba(
                    validation[features]
                )[:, 1]
                threshold = select_threshold(
                    validation["target_event"], validation_probability
                )

                final_model = step7.fit_boosting(development, features)
                test_probability = final_model.predict_proba(
                    test[features]
                )[:, 1]
                test_prediction = (
                    test_probability >= threshold
                ).astype(int)
                metric_rows.append({
                    "marker": marker,
                    "direction": direction,
                    "test_year": test_year,
                    "feature_block": block,
                    "n_features": len(features),
                    "selected_threshold": threshold,
                    **complete_metrics(
                        test["target_event"],
                        test_probability,
                        prediction=test_prediction,
                    ),
                })
                prediction_rows.append(pd.DataFrame({
                    "person_key": test["person_key"].to_numpy(),
                    "GV_KEY": test["GV_KEY"].to_numpy(),
                    "year": test["year"].to_numpy(),
                    "next_year": test["next_year"].to_numpy(),
                    "marker": marker,
                    "direction": direction,
                    "feature_block": block,
                    "target_event": test["target_event"].to_numpy(),
                    "probability": test_probability,
                    "prediction": test_prediction,
                    "selected_threshold": threshold,
                }))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    yearly = pd.DataFrame(metric_rows)
    pooled_rows = []
    for (marker, direction, block), group in predictions.groupby(
        ["marker", "direction", "feature_block"], sort=True
    ):
        pooled_rows.append({
            "marker": marker,
            "direction": direction,
            "feature_block": block,
            "n_features": int(
                yearly[
                    (yearly["marker"] == marker)
                    & (yearly["direction"] == direction)
                    & (yearly["feature_block"] == block)
                ]["n_features"].iloc[0]
            ),
            **complete_metrics(
                group["target_event"],
                group["probability"],
                prediction=group["prediction"],
            ),
        })
    return yearly, pd.DataFrame(pooled_rows), pd.DataFrame(feature_rows)


def empty_ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    note = pd.DataFrame({
        "note": [
            "Ablation skipped because STEP8_SKIP_ABLATION=1. "
            "Run without this variable for the article-grade ablation."
        ]
    })
    return note.copy(), note.copy(), note.copy()


def write_results(
    selected: pd.DataFrame,
    selection: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    yearly_stability: pd.DataFrame,
    stability_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    ablation_yearly: pd.DataFrame,
    ablation_pooled: pd.DataFrame,
    ablation_features: pd.DataFrame,
) -> None:
    progress(f"STEP 8.7/7: saving {OUTPUT_FILE.name}...")
    readme = pd.DataFrame({
        "item": [
            "Purpose",
            "Temporal selection",
            "First test year",
            "Uncertainty",
            "Paired comparison",
            "Primary metric",
            "Ablation A",
            "Ablation B",
            "Ablation C",
            "Ablation D",
            "Causal limitation",
        ],
        "definition": [
            "Article-grade validation of the Step 7 marker models.",
            "For test year t, the branch is selected only from out-of-sample performance in t-1.",
            "The earliest year uses prespecified base_boosting; it is not selected on test labels.",
            f"95% percentile intervals from {N_BOOTSTRAPS} company-cluster bootstrap samples.",
            "Candidate-minus-base differences use identical resampled companies.",
            "PR-AUC relative to event prevalence; ROC-AUC is secondary.",
            "Current marker position only.",
            "A plus individual transition history.",
            "B plus current and lagged company context.",
            "C plus target-state tenure and lagged change dynamics.",
            "Predictive associations do not establish that one person's change causes another's.",
        ],
    })

    safe_prediction_columns = [
        "person_key", "GV_KEY", "year", "next_year", "target_event",
        "marker", "direction", "selected_model", "selected_probability",
        "selected_class", "selection_year", "selection_basis",
    ]
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="readme", index=False)
        selected_metrics.to_excel(
            writer, sheet_name="selected_strategy", index=False
        )
        selection.to_excel(
            writer, sheet_name="model_selection_audit", index=False
        )
        confidence_intervals.to_excel(
            writer, sheet_name="bootstrap_ci", index=False
        )
        paired_comparisons.to_excel(
            writer, sheet_name="paired_model_tests", index=False
        )
        yearly_stability.to_excel(
            writer, sheet_name="yearly_stability", index=False
        )
        stability_summary.to_excel(
            writer, sheet_name="stability_summary", index=False
        )
        importance_summary.to_excel(
            writer, sheet_name="importance_summary", index=False
        )
        ablation_pooled.to_excel(
            writer, sheet_name="ablation_pooled", index=False
        )
        ablation_yearly.to_excel(
            writer, sheet_name="ablation_yearly", index=False
        )
        ablation_features.to_excel(
            writer, sheet_name="ablation_features", index=False
        )
        selected[safe_prediction_columns].to_excel(
            writer, sheet_name="selected_predictions", index=False
        )


def print_summary(
    selected_metrics: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    ablation_pooled: pd.DataFrame,
) -> None:
    print("\nPREQUENTIAL SELECTED STRATEGY", flush=True)
    pooled = selected_metrics[selected_metrics["test_year"].isna()].copy()
    columns = [
        "marker", "direction", "pr_auc", "pr_auc_over_prevalence",
        "roc_auc", "top_10_lift", "precision", "recall",
    ]
    print(pooled[columns].round(4).to_string(index=False), flush=True)

    print("\n95% CLUSTER BOOTSTRAP INTERVALS: SELECTED STRATEGY", flush=True)
    ci = confidence_intervals[
        confidence_intervals["model"] == "prequential_selected"
    ]
    print(
        ci[[
            "marker", "direction", "metric", "estimate",
            "ci_95_low", "ci_95_high",
        ]].round(4).to_string(index=False),
        flush=True,
    )

    print("\nPAIRED DIFFERENCE: SELECTED STRATEGY MINUS BASE", flush=True)
    paired = paired_comparisons[
        paired_comparisons["candidate_model"] == "prequential_selected"
    ]
    print(
        paired[[
            "marker", "direction", "metric", "difference",
            "difference_ci_95_low", "difference_ci_95_high",
            "two_sided_bootstrap_p",
        ]].round(4).to_string(index=False),
        flush=True,
    )

    if "feature_block" in ablation_pooled.columns:
        print("\nPOOLED FEATURE ABLATION", flush=True)
        print(
            ablation_pooled[[
                "marker", "direction", "feature_block", "n_features",
                "pr_auc", "roc_auc", "top_10_lift",
            ]].round(4).to_string(index=False),
            flush=True,
        )


def main() -> None:
    total_started = perf_counter()
    predictions, importance = load_step7_predictions()
    if SMOKE_TEST:
        first_target = predictions[["marker", "direction"]].drop_duplicates().iloc[0]
        predictions = predictions[
            (predictions["marker"] == first_target["marker"])
            & (predictions["direction"] == first_target["direction"])
        ].copy()
        importance = importance[
            (importance["marker"] == first_target["marker"])
            & (importance["direction"] == first_target["direction"])
        ].copy()

    selected, selection, selected_metrics = build_prequential_selection(
        predictions
    )
    confidence_intervals, paired_comparisons = (
        bootstrap_confidence_intervals(selected)
    )
    yearly_stability, stability_summary = build_stability_tables(selected)
    importance_summary = aggregate_importance(importance)

    if SKIP_ABLATION:
        progress("STEP 8.6/7: ablation skipped by STEP8_SKIP_ABLATION=1.")
        ablation_yearly, ablation_pooled, ablation_features = (
            empty_ablation_tables()
        )
    else:
        ablation_yearly, ablation_pooled, ablation_features = run_ablation()

    write_results(
        selected,
        selection,
        selected_metrics,
        confidence_intervals,
        paired_comparisons,
        yearly_stability,
        stability_summary,
        importance_summary,
        ablation_yearly,
        ablation_pooled,
        ablation_features,
    )
    print_summary(
        selected_metrics,
        confidence_intervals,
        paired_comparisons,
        ablation_pooled,
    )
    elapsed = perf_counter() - total_started
    progress(
        f"STEP 8 COMPLETE. Saved to {OUTPUT_FILE.resolve()}. "
        f"Total time: {elapsed / 60:.1f} min."
    )


if __name__ == "__main__":
    # Imported only here so the fast statistical part remains readable and
    # Step 7's feature engineering/model configuration is reused exactly.
    import person_step_7_gower_hybrid as step7

    main()
