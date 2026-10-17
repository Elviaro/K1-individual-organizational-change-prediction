"""Step 7: enhanced temporal model, Gower precedents and hybrid forecast.

Put this script next to:
  * person_step_2_transitions.xlsx
  * 20260721_OD_Database_1993_2024b.xlsx

Run:
    python person_step_7_gower_hybrid.py

For a quick technical test (one target, one test year):
    $env:STEP7_SMOKE = "1"
    python person_step_7_gower_hybrid.py

The main analysis is conditional on a person being observed in two adjacent
years, because Step 2 contains such transitions.  All predictors are measured
at year t or earlier; next-year marker values are used only as outcomes.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSITIONS_FILE = Path("person_step_2_transitions.xlsx")
RAW_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("person_step_7_gower_hybrid_result.xlsx")

RANDOM_STATE = 42
SMOKE_TEST = os.getenv("STEP7_SMOKE", "0") == "1"
TEST_YEARS = [2023] if SMOKE_TEST else [2019, 2020, 2021, 2022, 2023]

MARKERS = [
    "vp", "svp", "evp", "sevp", "dir", "sdir", "md", "smd", "se",
    "vc", "svc", "president", "board", "ceo", "cxo", "primary",
    "support", "bu",
]

TARGETS = [
    ("evp", "onset"),
    ("support", "loss"),
    ("primary", "onset"),
    ("bu", "loss"),
    ("cxo", "loss"),
]
if SMOKE_TEST:
    TARGETS = TARGETS[:1]

MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 180,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

GOWER_K_VALUES = [25, 50, 100]
GOWER_MAX_REFERENCE = 2500 if SMOKE_TEST else 18000
GOWER_BLOCK_SIZE = 256
N_STORED_NEIGHBORS = 3

TEXT_MAX_FEATURES = 2000 if SMOKE_TEST else 12000
TEXT_MIN_DF = 3

HYBRID_WEIGHTS = [
    (1.00, 0.00, 0.00),
    (0.75, 0.25, 0.00),
    (0.75, 0.00, 0.25),
    (0.60, 0.20, 0.20),
    (0.50, 0.30, 0.20),
    (0.50, 0.25, 0.25),
]


def progress(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [TRANSITIONS_FILE, RAW_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path.resolve()}"
            )

    progress("STEP 7.1/8: loading transition and raw tables...")
    transitions = pd.read_excel(
        TRANSITIONS_FILE,
        sheet_name="person_transitions",
    )
    raw_columns = [
        "year", "company", "GV_KEY", "full_name", "role", "TMTSource",
        "GICSectors", *MARKERS,
    ]
    raw = pd.read_excel(RAW_FILE, sheet_name=0, usecols=raw_columns)

    required = {
        "person_key", "year", "next_year", "GV_KEY", "full_name",
        *MARKERS, *[f"next_{marker}" for marker in MARKERS],
    }
    missing = sorted(required.difference(transitions.columns))
    if missing:
        raise ValueError("Missing transition columns: " + ", ".join(missing))

    transitions["year"] = pd.to_numeric(transitions["year"]).astype(int)
    transitions["next_year"] = pd.to_numeric(
        transitions["next_year"]
    ).astype(int)
    transitions["GV_KEY"] = pd.to_numeric(
        transitions["GV_KEY"]
    ).astype("int64")
    if ((transitions["next_year"] - transitions["year"]) != 1).any():
        raise ValueError("Step 2 contains non-adjacent transitions.")

    raw = raw[raw["year"].notna() & raw["GV_KEY"].notna()].copy()
    raw["year"] = pd.to_numeric(raw["year"]).astype(int)
    raw["GV_KEY"] = pd.to_numeric(raw["GV_KEY"]).astype("int64")
    raw["person_key"] = (
        raw["GV_KEY"].astype(str)
        + "|"
        + raw["full_name"].map(normalize_name)
    )
    for marker in MARKERS:
        transitions[marker] = pd.to_numeric(
            transitions[marker], errors="coerce"
        ).fillna(0).astype("int8")
        transitions[f"next_{marker}"] = pd.to_numeric(
            transitions[f"next_{marker}"], errors="coerce"
        ).fillna(0).astype("int8")
        raw[marker] = pd.to_numeric(
            raw[marker], errors="coerce"
        ).fillna(0).clip(0, 1).astype("int8")

    progress(
        f"Loaded {len(transitions):,} transitions and "
        f"{len(raw):,} raw person-year rows."
    )
    return transitions, raw


def build_raw_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build current-year company features from the full observed roster."""
    progress("STEP 7.2/8: building leakage-safe raw company features...")

    person_aggregation = {marker: "max" for marker in MARKERS}
    person_aggregation.update({
        "company": "first",
        "role": lambda x: " | ".join(
            sorted(set(x.dropna().astype(str)))
        ),
        "TMTSource": lambda x: "|".join(
            sorted(set(x.dropna().astype(str)))
        ),
        "GICSectors": "first",
    })
    raw_person = (
        raw.groupby(["person_key", "GV_KEY", "year"], as_index=False)
        .agg(person_aggregation)
    )

    groups = raw_person.groupby(["GV_KEY", "year"], sort=False)
    raw_person["raw_company_tmt_size"] = groups["person_key"].transform(
        "size"
    )
    company_features = ["raw_company_tmt_size"]
    for marker in MARKERS:
        count = f"raw_company_{marker}_count"
        share = f"raw_company_{marker}_share"
        raw_person[count] = groups[marker].transform("sum")
        raw_person[share] = raw_person[count] / raw_person[
            "raw_company_tmt_size"
        ]
        company_features.extend([count, share])

    raw_person["raw_profile_code"] = (
        raw_person[MARKERS].astype(str).agg("".join, axis=1)
    )
    raw_person["raw_profile_company_count"] = (
        raw_person.groupby(
            ["GV_KEY", "year", "raw_profile_code"], sort=False
        )["person_key"].transform("size")
    )
    raw_person["raw_profile_company_share"] = (
        raw_person["raw_profile_company_count"]
        / raw_person["raw_company_tmt_size"]
    )
    company_features.extend([
        "raw_profile_company_count",
        "raw_profile_company_share",
    ])

    # Distance from the current company-year marker centroid.
    centroid_columns = []
    for marker in MARKERS:
        deviation = f"raw_{marker}_peer_deviation"
        raw_person[deviation] = (
            raw_person[marker]
            - raw_person[f"raw_company_{marker}_share"]
        ).abs()
        centroid_columns.append(deviation)
    raw_person["raw_mean_peer_deviation"] = raw_person[
        centroid_columns
    ].mean(axis=1)
    company_features.append("raw_mean_peer_deviation")

    keep = [
        "person_key", "GV_KEY", "year", "role", "TMTSource",
        "GICSectors", *company_features,
    ]
    return raw_person[keep], company_features


def consecutive_tenure(data: pd.DataFrame, marker: str) -> pd.Series:
    previous_value = data.groupby("person_key", sort=False)[marker].shift(1)
    previous_year = data.groupby("person_key", sort=False)["year"].shift(1)
    new_run = ~(
        data[marker].eq(previous_value)
        & data["year"].eq(previous_year + 1)
    )
    run_id = new_run.groupby(data["person_key"]).cumsum()
    return data.groupby([data["person_key"], run_id]).cumcount() + 1


def prepare_common_features(
    transitions: pd.DataFrame,
    raw_features: pd.DataFrame,
    raw_company_columns: list[str],
) -> pd.DataFrame:
    progress("STEP 7.3/8: engineering historical person/company features...")
    result = transitions.sort_values(
        ["person_key", "year"]
    ).reset_index(drop=True).copy()

    result["current_marker_count"] = result[MARKERS].sum(axis=1)
    result["multiple_current_markers"] = (
        result["current_marker_count"] > 1
    ).astype("int8")
    next_markers = [f"next_{marker}" for marker in MARKERS]
    result["any_change_current_transition"] = (
        result[MARKERS].to_numpy() != result[next_markers].to_numpy()
    ).any(axis=1).astype("int8")
    result["n_changes_current_transition"] = (
        result[MARKERS].to_numpy() != result[next_markers].to_numpy()
    ).sum(axis=1).astype("int8")

    people = result.groupby("person_key", sort=False)
    result["prior_transition_count"] = people.cumcount()
    result["years_since_first_observation"] = (
        result["year"] - people["year"].transform("min")
    )
    result["prior_any_change_count"] = (
        people["any_change_current_transition"].cumsum()
        - result["any_change_current_transition"]
    )
    result["prior_any_change_rate"] = np.where(
        result["prior_transition_count"] > 0,
        result["prior_any_change_count"]
        / result["prior_transition_count"],
        0.0,
    )
    result["prior_changed_marker_count"] = (
        people["n_changes_current_transition"].cumsum()
        - result["n_changes_current_transition"]
    )
    result["prior_mean_markers_changed"] = np.where(
        result["prior_transition_count"] > 0,
        result["prior_changed_marker_count"]
        / result["prior_transition_count"],
        0.0,
    )
    previous_year = people["year"].shift(1)
    adjacent = previous_year.eq(result["year"] - 1)
    result["previous_transition_changed"] = (
        people["any_change_current_transition"].shift(1).fillna(0)
    ).where(adjacent, 0).astype("int8")
    result["previous_transition_change_count"] = (
        people["n_changes_current_transition"].shift(1).fillna(0)
    ).where(adjacent, 0).astype("int8")
    result["previous_transition_is_adjacent"] = adjacent.astype("int8")

    # Completed company transitions ending at current year are legitimate
    # lagged predictors for the transition current year -> next year.
    lagged_company = (
        result.groupby(["GV_KEY", "next_year"], as_index=False)
        .agg(
            lag_company_eligible=("person_key", "size"),
            lag_company_any_changes=("any_change_current_transition", "sum"),
            lag_company_marker_changes=("n_changes_current_transition", "sum"),
        )
        .rename(columns={"next_year": "year"})
    )
    lagged_company["lag_company_any_change_rate"] = (
        lagged_company["lag_company_any_changes"]
        / lagged_company["lag_company_eligible"]
    )
    lagged_company["lag_company_mean_marker_changes"] = (
        lagged_company["lag_company_marker_changes"]
        / lagged_company["lag_company_eligible"]
    )
    result = result.merge(
        lagged_company,
        on=["GV_KEY", "year"],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        raw_features,
        on=["person_key", "GV_KEY", "year"],
        how="left",
        validate="many_to_one",
    )
    result["role"] = result["role"].fillna("").astype(str)
    result["GICSectors"] = result["GICSectors"].fillna("unknown").astype(str)
    numeric_fill = [
        *raw_company_columns,
        "lag_company_eligible",
        "lag_company_any_changes",
        "lag_company_marker_changes",
        "lag_company_any_change_rate",
        "lag_company_mean_marker_changes",
    ]
    result[numeric_fill] = result[numeric_fill].fillna(0)
    return result


def build_target_data(
    common: pd.DataFrame,
    marker: str,
    direction: str,
    raw_company_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    required_state = 0 if direction == "onset" else 1
    population = common[common[marker] == required_state].copy()
    population["target_event"] = (
        population[f"next_{marker}"]
        if direction == "onset"
        else 1 - population[f"next_{marker}"]
    ).astype("int8")

    common = common.copy()
    common["target_history_event"] = np.where(
        direction == "onset",
        (common[marker] == 0) & (common[f"next_{marker}"] == 1),
        (common[marker] == 1) & (common[f"next_{marker}"] == 0),
    ).astype("int8")
    common["target_state_tenure"] = consecutive_tenure(common, marker)
    history = common[[
        "person_key", "year", "target_history_event", "target_state_tenure",
    ]].sort_values(["person_key", "year"])
    grouped = history.groupby("person_key", sort=False)
    history["prior_target_event_count"] = (
        grouped["target_history_event"].cumsum()
        - history["target_history_event"]
    )
    previous_year = grouped["year"].shift(1)
    history["previous_target_event"] = (
        grouped["target_history_event"].shift(1).fillna(0)
    ).where(previous_year.eq(history["year"] - 1), 0).astype("int8")
    population = population.merge(
        history[[
            "person_key", "year", "target_state_tenure",
            "prior_target_event_count", "previous_target_event",
        ]],
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

    event_source = common[common[marker] == required_state].copy()
    company_history = (
        event_source.groupby(["GV_KEY", "next_year"], as_index=False)
        .agg(
            company_prior_target_eligible=("target_history_event", "size"),
            company_prior_target_events=("target_history_event", "sum"),
        )
        .rename(columns={"next_year": "year"})
    )
    company_history["company_prior_target_rate"] = (
        company_history["company_prior_target_events"]
        / company_history["company_prior_target_eligible"]
    )
    population = population.merge(
        company_history,
        on=["GV_KEY", "year"],
        how="left",
        validate="many_to_one",
    )
    history_columns = [
        "company_prior_target_eligible",
        "company_prior_target_events",
        "company_prior_target_rate",
    ]
    population[history_columns] = population[history_columns].fillna(0)

    other_markers = [item for item in MARKERS if item != marker]
    base_features = [
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
        "company_prior_target_eligible",
        "company_prior_target_events",
        "company_prior_target_rate",
        *raw_company_columns,
    ]
    enhanced_additions = [
        "target_state_tenure",
        "prior_changed_marker_count",
        "prior_mean_markers_changed",
        "previous_transition_change_count",
        "lag_company_eligible",
        "lag_company_any_changes",
        "lag_company_marker_changes",
        "lag_company_any_change_rate",
        "lag_company_mean_marker_changes",
    ]
    enhanced_features = list(dict.fromkeys(base_features + enhanced_additions))
    population[enhanced_features] = (
        population[enhanced_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return population, base_features, enhanced_features


def split_fold(data: pd.DataFrame, test_year: int):
    validation_year = test_year - 1
    train = data[data["next_year"] < validation_year].copy()
    validation = data[data["next_year"] == validation_year].copy()
    test = data[data["next_year"] == test_year].copy()
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty or part["target_event"].nunique() < 2:
            raise ValueError(f"Invalid {name} fold for target year {test_year}.")
    return train, validation, test


def fit_boosting(data: pd.DataFrame, features: list[str]):
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    weights = compute_sample_weight("balanced", data["target_event"])
    model.fit(data[features], data["target_event"], sample_weight=weights)
    return model


def fit_text_model(train: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=TEXT_MIN_DF,
        max_features=TEXT_MAX_FEATURES,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(train["role"].fillna(""))
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    model.fit(matrix, train["target_event"])
    return vectorizer, model


def text_probability(vectorizer, model, data: pd.DataFrame) -> np.ndarray:
    matrix = vectorizer.transform(data["role"].fillna(""))
    return model.predict_proba(matrix)[:, 1]


def choose_reference(train: pd.DataFrame) -> pd.DataFrame:
    positives = train[train["target_event"] == 1]
    negatives = train[train["target_event"] == 0]
    remaining = max(GOWER_MAX_REFERENCE - len(positives), 1000)
    if len(negatives) > remaining:
        negatives = negatives.sample(remaining, random_state=RANDOM_STATE)
    reference = pd.concat([positives, negatives], ignore_index=True)
    return reference.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def gower_knn_multi(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    features: list[str],
    k_values: list[int],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Compute several k values in one distance pass."""
    ref_x = reference[features].to_numpy(dtype=np.float32)
    query_x = query[features].to_numpy(dtype=np.float32)
    ref_y = reference["target_event"].to_numpy(dtype=np.float32)
    minimum = np.nanmin(ref_x, axis=0)
    span = np.nanmax(ref_x, axis=0) - minimum
    span[span == 0] = 1.0
    ref_scaled = (ref_x - minimum) / span
    query_scaled = (query_x - minimum) / span
    effective_k = sorted({min(int(k), len(reference)) for k in k_values})
    max_k = max(effective_k)

    probabilities = {
        k: np.empty(len(query), dtype=float) for k in effective_k
    }
    stored_indices = np.empty((len(query), N_STORED_NEIGHBORS), dtype=int)
    stored_distances = np.empty((len(query), N_STORED_NEIGHBORS), dtype=float)
    for start in range(0, len(query), GOWER_BLOCK_SIZE):
        stop = min(start + GOWER_BLOCK_SIZE, len(query))
        # Accumulate one feature at a time.  This avoids allocating a very
        # large query x reference x feature tensor.
        distances = np.zeros(
            (stop - start, len(reference)), dtype=np.float32
        )
        query_block = query_scaled[start:stop]
        for feature_index in range(len(features)):
            distances += np.abs(
                query_block[:, None, feature_index]
                - ref_scaled[None, :, feature_index]
            )
        distances /= len(features)
        neighbor_indices = np.argpartition(
            distances, kth=max_k - 1, axis=1
        )[:, :max_k]
        neighbor_distances = np.take_along_axis(
            distances, neighbor_indices, axis=1
        )
        full_order = np.argsort(neighbor_distances, axis=1)
        neighbor_indices = np.take_along_axis(
            neighbor_indices, full_order, axis=1
        )
        neighbor_distances = np.take_along_axis(
            neighbor_distances, full_order, axis=1
        )
        weights = 1.0 / (neighbor_distances + 0.02)
        weighted_outcomes = weights * ref_y[neighbor_indices]
        cumulative_weight = np.cumsum(weights, axis=1)
        cumulative_outcome = np.cumsum(weighted_outcomes, axis=1)
        for k in effective_k:
            probabilities[k][start:stop] = (
                cumulative_outcome[:, k - 1] / cumulative_weight[:, k - 1]
            )
        nearest = neighbor_indices[:, :N_STORED_NEIGHBORS]
        nearest_dist = neighbor_distances[:, :N_STORED_NEIGHBORS]
        stored_indices[start:stop] = nearest
        stored_distances[start:stop] = nearest_dist
    return probabilities, stored_indices, stored_distances


def gower_knn(reference, query, features, k):
    probabilities, indices, distances = gower_knn_multi(
        reference, query, features, [k]
    )
    effective_k = min(int(k), len(reference))
    return probabilities[effective_k], indices, distances


def safe_roc_auc(y_true, probability):
    return float(roc_auc_score(y_true, probability))


def risk_metrics(y_true, probability, share=0.10):
    y = np.asarray(y_true, dtype=int)
    n = max(1, int(np.ceil(len(y) * share)))
    selected = np.argsort(-np.asarray(probability))[:n]
    precision = float(y[selected].mean())
    prevalence = float(y.mean())
    return {
        "top_10_precision": precision,
        "top_10_lift": precision / prevalence if prevalence else np.nan,
        "top_10_event_capture": (
            float(y[selected].sum() / y.sum()) if y.sum() else np.nan
        ),
    }


def select_threshold(y_true, probability):
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = np.lexsort((thresholds, recall, precision, f1))[-1]
    return float(thresholds[best])


def complete_metrics(y_true, probability, threshold=None, prediction=None):
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if prediction is None:
        prediction = (probability >= threshold).astype(int)
    else:
        prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, probability))
    result = {
        "n": len(y),
        "n_positive": int(y.sum()),
        "positive_share": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_over_prevalence": pr_auc / prevalence,
        "roc_auc": safe_roc_auc(y, probability),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    result.update(risk_metrics(y, probability))
    return result


def select_gower_k(reference, validation, features):
    rows, cache = [], {}
    probabilities, indices, distances = gower_knn_multi(
        reference, validation, features, GOWER_K_VALUES
    )
    for k in GOWER_K_VALUES:
        effective_k = min(k, len(reference))
        probability = probabilities[effective_k]
        cache[k] = (probability, indices, distances)
        rows.append({
            "k": k,
            "validation_pr_auc": average_precision_score(
                validation["target_event"], probability
            ),
            "validation_roc_auc": roc_auc_score(
                validation["target_event"], probability
            ),
        })
    table = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    best_k = int(table.iloc[0]["k"])
    return best_k, cache[best_k], table


def select_hybrid(y_true, enhanced, gower, text):
    rows = []
    for boost_weight, gower_weight, text_weight in HYBRID_WEIGHTS:
        probability = (
            boost_weight * enhanced
            + gower_weight * gower
            + text_weight * text
        )
        rows.append({
            "boost_weight": boost_weight,
            "gower_weight": gower_weight,
            "text_weight": text_weight,
            "validation_pr_auc": average_precision_score(y_true, probability),
            "validation_roc_auc": roc_auc_score(y_true, probability),
        })
    table = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    )
    best = table.iloc[0]
    weights = (
        float(best["boost_weight"]),
        float(best["gower_weight"]),
        float(best["text_weight"]),
    )
    return weights, table


def run_fold(population, base_features, enhanced_features, marker, direction, test_year):
    progress(f"  {marker} {direction}, test {test_year}: splitting data...")
    train, validation, test = split_fold(population, test_year)

    base_model = fit_boosting(train, base_features)
    enhanced_model = fit_boosting(train, enhanced_features)
    vectorizer, text_model = fit_text_model(train)

    base_val = base_model.predict_proba(validation[base_features])[:, 1]
    enhanced_val = enhanced_model.predict_proba(
        validation[enhanced_features]
    )[:, 1]
    text_val = text_probability(vectorizer, text_model, validation)

    reference = choose_reference(train)
    best_k, gower_validation_result, k_table = select_gower_k(
        reference, validation, enhanced_features
    )
    gower_val = gower_validation_result[0]
    weights, hybrid_table = select_hybrid(
        validation["target_event"], enhanced_val, gower_val, text_val
    )
    boost_weight, gower_weight, text_weight = weights
    hybrid_val = (
        boost_weight * enhanced_val
        + gower_weight * gower_val
        + text_weight * text_val
    )

    thresholds = {
        "base_boosting": select_threshold(validation["target_event"], base_val),
        "enhanced_boosting": select_threshold(
            validation["target_event"], enhanced_val
        ),
        "gower_precedents": select_threshold(
            validation["target_event"], gower_val
        ),
        "role_text": select_threshold(validation["target_event"], text_val),
        "hybrid": select_threshold(validation["target_event"], hybrid_val),
    }

    development = pd.concat([train, validation], ignore_index=True)
    base_model = fit_boosting(development, base_features)
    enhanced_model = fit_boosting(development, enhanced_features)
    vectorizer, text_model = fit_text_model(development)
    reference = choose_reference(development)

    base_test = base_model.predict_proba(test[base_features])[:, 1]
    enhanced_test = enhanced_model.predict_proba(
        test[enhanced_features]
    )[:, 1]
    text_test = text_probability(vectorizer, text_model, test)
    gower_test, neighbor_indices, neighbor_distances = gower_knn(
        reference, test, enhanced_features, best_k
    )
    hybrid_test = (
        boost_weight * enhanced_test
        + gower_weight * gower_test
        + text_weight * text_test
    )

    probabilities = {
        "base_boosting": base_test,
        "enhanced_boosting": enhanced_test,
        "gower_precedents": gower_test,
        "role_text": text_test,
        "hybrid": hybrid_test,
    }
    metric_rows = []
    for model_name, probability in probabilities.items():
        metric_rows.append({
            "marker": marker,
            "direction": direction,
            "target_year": test_year,
            "model": model_name,
            "selected_threshold": thresholds[model_name],
            "selected_k": best_k if "gower" in model_name or model_name == "hybrid" else np.nan,
            "boost_weight": boost_weight if model_name == "hybrid" else np.nan,
            "gower_weight": gower_weight if model_name == "hybrid" else np.nan,
            "text_weight": text_weight if model_name == "hybrid" else np.nan,
            **complete_metrics(
                test["target_event"], probability, thresholds[model_name]
            ),
        })

    predictions = test[[
        "person_key", "company", "GV_KEY", "full_name", "role",
        "year", "next_year", "target_event",
    ]].copy()
    predictions["marker"] = marker
    predictions["direction"] = direction
    for name, probability in probabilities.items():
        predictions[f"probability_{name}"] = probability
        predictions[f"threshold_{name}"] = thresholds[name]
        predictions[f"class_{name}"] = (
            probability >= thresholds[name]
        ).astype("int8")
    predictions["hybrid_threshold"] = thresholds["hybrid"]
    predictions["hybrid_prediction"] = (
        hybrid_test >= thresholds["hybrid"]
    ).astype("int8")

    for rank in range(N_STORED_NEIGHBORS):
        neighbor_rows = reference.iloc[neighbor_indices[:, rank]]
        predictions[f"neighbor_{rank + 1}_person_key"] = neighbor_rows[
            "person_key"
        ].to_numpy()
        predictions[f"neighbor_{rank + 1}_year"] = neighbor_rows[
            "year"
        ].to_numpy()
        predictions[f"neighbor_{rank + 1}_outcome"] = neighbor_rows[
            "target_event"
        ].to_numpy()
        predictions[f"neighbor_{rank + 1}_distance"] = neighbor_distances[
            :, rank
        ]

    # Model-agnostic permutation importance for the enhanced branch.
    importance_sample = test
    if len(test) > 2000:
        importance_sample = test.sample(2000, random_state=RANDOM_STATE)
    importance = permutation_importance(
        enhanced_model,
        importance_sample[enhanced_features],
        importance_sample["target_event"],
        scoring="average_precision",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame({
        "marker": marker,
        "direction": direction,
        "target_year": test_year,
        "feature": enhanced_features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    k_table.insert(0, "marker", marker)
    k_table.insert(1, "direction", direction)
    k_table.insert(2, "target_year", test_year)
    hybrid_table.insert(0, "marker", marker)
    hybrid_table.insert(1, "direction", direction)
    hybrid_table.insert(2, "target_year", test_year)
    return metric_rows, predictions, k_table, hybrid_table, importance_table


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    probability_columns = [
        column for column in predictions.columns
        if column.startswith("probability_")
    ]
    for (marker, direction), group in predictions.groupby(
        ["marker", "direction"]
    ):
        for column in probability_columns:
            model_name = column.replace("probability_", "")
            probability = group[column].to_numpy()
            rows.append({
                "marker": marker,
                "direction": direction,
                "model": model_name,
                **complete_metrics(
                    group["target_event"],
                    probability,
                    prediction=group[f"class_{model_name}"],
                ),
            })
    return pd.DataFrame(rows)


def save_results(metrics, predictions, k_tables, hybrid_tables, importance):
    progress(f"STEP 7.8/8: saving {OUTPUT_FILE.name}...")
    yearly = pd.DataFrame(metrics)
    all_predictions = pd.concat(predictions, ignore_index=True)
    pooled = pooled_metrics(all_predictions)
    comparison = (
        pooled.pivot_table(
            index=["marker", "direction"],
            columns="model",
            values=["pr_auc", "roc_auc", "top_10_lift"],
        )
        .reset_index()
    )
    comparison.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple) else column
        for column in comparison.columns
    ]

    readme = pd.DataFrame({
        "item": [
            "Population", "Temporal design", "Base branch", "Enhanced branch",
            "Precedent branch", "Text branch", "Hybrid selection", "Main metric",
        ],
        "definition": [
            "People observed in the same company in adjacent years.",
            "Expanding training window, previous-year validation, next-year test.",
            "Historical gradient boosting using current and lagged predictors.",
            "Boosting plus target-state tenure and lagged company dynamics.",
            "Approximate Gower kNN using only historical reference cases.",
            "TF-IDF role-title logistic regression.",
            "Weights selected on validation PR-AUC only.",
            "PR-AUC and PR-AUC/prevalence; ROC-AUC and top-decile lift are secondary.",
        ],
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="readme", index=False)
        comparison.to_excel(writer, sheet_name="model_comparison", index=False)
        pooled.to_excel(writer, sheet_name="pooled_metrics", index=False)
        yearly.to_excel(writer, sheet_name="yearly_metrics", index=False)
        pd.concat(k_tables, ignore_index=True).to_excel(
            writer, sheet_name="gower_k_selection", index=False
        )
        pd.concat(hybrid_tables, ignore_index=True).to_excel(
            writer, sheet_name="hybrid_selection", index=False
        )
        pd.concat(importance, ignore_index=True).to_excel(
            writer, sheet_name="permutation_importance", index=False
        )
        all_predictions.to_excel(
            writer, sheet_name="test_predictions", index=False
        )
    return pooled, comparison


def main() -> None:
    started = perf_counter()
    transitions, raw = load_inputs()
    raw_features, raw_company_columns = build_raw_features(raw)
    common = prepare_common_features(
        transitions, raw_features, raw_company_columns
    )

    all_metrics, all_predictions = [], []
    all_k_tables, all_hybrid_tables, all_importance = [], [], []
    progress("STEP 7.4/8: running target-specific rolling experiments...")
    for target_number, (marker, direction) in enumerate(TARGETS, start=1):
        progress(
            f"TARGET {target_number}/{len(TARGETS)}: {marker} {direction}"
        )
        population, base_features, enhanced_features = build_target_data(
            common, marker, direction, raw_company_columns
        )
        for year_number, test_year in enumerate(TEST_YEARS, start=1):
            progress(
                f"  rolling fold {year_number}/{len(TEST_YEARS)}: {test_year}"
            )
            result = run_fold(
                population, base_features, enhanced_features,
                marker, direction, test_year,
            )
            metrics, predictions, k_table, hybrid_table, importance = result
            all_metrics.extend(metrics)
            all_predictions.append(predictions)
            all_k_tables.append(k_table)
            all_hybrid_tables.append(hybrid_table)
            all_importance.append(importance)

    progress("STEP 7.7/8: pooling strictly out-of-sample predictions...")
    pooled, comparison = save_results(
        all_metrics, all_predictions, all_k_tables,
        all_hybrid_tables, all_importance,
    )
    print("\nPOOLED MODEL RESULTS", flush=True)
    print(
        pooled[[
            "marker", "direction", "model", "n", "positive_share",
            "pr_auc", "pr_auc_over_prevalence", "roc_auc",
            "top_10_lift", "top_10_event_capture",
        ]].round(4).to_string(index=False),
        flush=True,
    )
    print("\nMODEL COMPARISON", flush=True)
    print(comparison.round(4).to_string(index=False), flush=True)
    progress(
        f"STEP 7 COMPLETE. Saved to {OUTPUT_FILE.resolve()}. "
        f"Total time: {(perf_counter() - started) / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
