"""Step 3B: enriched individual marker-change prediction.

Uses only information available at year t:
- current structural markers;
- current role title;
- prior history of the person;
- current company structure;
- company change history strictly before year t.

No next_*, delta_* or current-year outcomes of colleagues are used.
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


SOURCE_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
TRANSITIONS_FILE = Path("person_step_2_transitions.xlsx")
OUTPUT_FILE = Path("person_step_3_enriched_result.xlsx")

TRANSITIONS_SHEET = "person_transitions"
TARGET = "y_marker_set_changed"
RANDOM_STATE = 42

MARKERS = [
    "vp", "svp", "evp", "sevp", "dir", "sdir",
    "md", "smd", "se", "vc", "svc", "president",
    "board", "ceo", "cxo", "primary", "support", "bu",
]

GB_CANDIDATES = [
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 50,
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 50,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 50,
    },
]

ENSEMBLE_WEIGHTS = np.arange(0.0, 1.01, 0.1)
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def norm(value):
    if pd.isna(value):
        return ""

    value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9]", "", value)


def load_data():
    for path in [SOURCE_FILE, TRANSITIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path.resolve()}"
            )

    source = pd.read_excel(SOURCE_FILE)
    transitions = pd.read_excel(
        TRANSITIONS_FILE,
        sheet_name=TRANSITIONS_SHEET,
    )

    source_required = {
        "GV_KEY", "year", "full_name", "role", "uniqueid",
        *MARKERS,
    }
    transition_required = {
        "person_key", "year", "next_year", TARGET, *MARKERS,
    }

    missing_source = sorted(
        source_required - set(source.columns)
    )
    missing_transitions = sorted(
        transition_required - set(transitions.columns)
    )

    if missing_source:
        raise ValueError(
            "Missing source columns: "
            + ", ".join(missing_source)
        )

    if missing_transitions:
        raise ValueError(
            "Missing transition columns: "
            + ", ".join(missing_transitions)
        )

    source["year"] = pd.to_numeric(
        source["year"], errors="raise"
    ).astype(int)

    transitions["year"] = pd.to_numeric(
        transitions["year"], errors="raise"
    ).astype(int)

    transitions["next_year"] = pd.to_numeric(
        transitions["next_year"], errors="raise"
    ).astype(int)

    source["org_key"] = (
        source["GV_KEY"].astype("string").str.strip()
    )
    source["name_key"] = source["full_name"].map(norm)
    source["person_key"] = (
        source["org_key"].fillna("")
        + "|"
        + source["name_key"]
    )

    transitions["org_key"] = (
        transitions["person_key"]
        .astype("string")
        .str.split("|", n=1, regex=False)
        .str[0]
    )

    return source, transitions


def build_person_states(source):
    columns = [
        "person_key",
        "org_key",
        "year",
        "role",
        "full_name",
        *MARKERS,
    ]

    states = source[columns].copy()

    # Step 1 allowed only unique person-year states.
    unique = ~states.duplicated(
        ["person_key", "year"],
        keep=False,
    )
    states = states.loc[unique].copy()

    first_year = (
        states.groupby("person_key")["year"]
        .min()
        .rename("person_first_year")
    )

    states = states.merge(
        first_year,
        on="person_key",
        how="left",
        validate="many_to_one",
    )

    states["years_since_first_observation"] = (
        states["year"] - states["person_first_year"]
    )

    return states


def add_person_history(transitions):
    result = transitions.sort_values(
        ["person_key", "year"]
    ).copy()

    group = result.groupby(
        "person_key",
        sort=False,
    )

    result["prior_transition_count"] = group.cumcount()

    cumulative_changes = group[TARGET].cumsum()
    result["prior_change_count"] = (
        cumulative_changes - result[TARGET]
    )

    result["prior_change_rate"] = np.where(
        result["prior_transition_count"] > 0,
        result["prior_change_count"]
        / result["prior_transition_count"],
        0.0,
    )

    result["previous_transition_changed"] = (
        group[TARGET].shift(1).fillna(0).astype(int)
    )

    result["has_previous_transition"] = (
        result["prior_transition_count"] > 0
    ).astype(int)

    previous_year = group["year"].shift(1)
    result["previous_transition_was_adjacent"] = (
        previous_year.notna()
        & (result["year"] - previous_year).eq(1)
    ).astype(int)

    return result


def build_company_state_features(source):
    group = source.groupby(
        ["org_key", "year"],
        dropna=False,
    )

    company = group.agg(
        company_tmt_size=("uniqueid", "nunique"),
        company_role_count=("role", "nunique"),
    ).reset_index()

    marker_counts = (
        group[MARKERS]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                marker: f"company_{marker}_count"
                for marker in MARKERS
            }
        )
    )

    company = company.merge(
        marker_counts,
        on=["org_key", "year"],
        how="left",
        validate="one_to_one",
    )

    for marker in MARKERS:
        count_column = f"company_{marker}_count"
        share_column = f"company_{marker}_share"

        company[share_column] = (
            company[count_column]
            / company["company_tmt_size"].replace(0, np.nan)
        ).fillna(0.0)

    return company


def add_company_history(transitions):
    """Add company outcomes from years strictly before the current year."""

    yearly = (
        transitions.groupby(["org_key", "year"])[TARGET]
        .agg(
            company_year_changes="sum",
            company_year_transitions="size",
        )
        .reset_index()
        .sort_values(["org_key", "year"])
    )

    group = yearly.groupby(
        "org_key",
        sort=False,
    )

    cumulative_changes = group[
        "company_year_changes"
    ].cumsum()

    cumulative_transitions = group[
        "company_year_transitions"
    ].cumsum()

    yearly["company_prior_changes"] = (
        cumulative_changes
        - yearly["company_year_changes"]
    )

    yearly["company_prior_transitions"] = (
        cumulative_transitions
        - yearly["company_year_transitions"]
    )

    yearly["company_prior_change_rate"] = np.where(
        yearly["company_prior_transitions"] > 0,
        yearly["company_prior_changes"]
        / yearly["company_prior_transitions"],
        0.0,
    )

    history_columns = [
        "org_key",
        "year",
        "company_prior_changes",
        "company_prior_transitions",
        "company_prior_change_rate",
    ]

    return transitions.merge(
        yearly[history_columns],
        on=["org_key", "year"],
        how="left",
        validate="many_to_one",
    )


def build_features(source, transitions):
    result = add_person_history(transitions)

    person_states = build_person_states(source)

    person_columns = [
        "person_key",
        "year",
        "role",
        "person_first_year",
        "years_since_first_observation",
    ]

    result = result.merge(
        person_states[person_columns],
        on=["person_key", "year"],
        how="left",
        validate="one_to_one",
    )

    if result["role"].isna().any():
        n_missing = int(result["role"].isna().sum())
        raise ValueError(
            f"Role merge failed for {n_missing} transitions."
        )

    company_states = build_company_state_features(source)

    result = result.merge(
        company_states,
        on=["org_key", "year"],
        how="left",
        validate="many_to_one",
    )

    result = add_company_history(result)

    result["current_marker_count"] = (
        result[MARKERS].sum(axis=1)
    )

    result["multiple_current_markers"] = (
        result["current_marker_count"] > 1
    ).astype(int)

    result["role_text"] = (
        result["role"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return result


def split_by_time(data):
    years = sorted(data["year"].unique())

    if len(years) < 3:
        raise ValueError(
            "At least three transition years are required."
        )

    validation_year = years[-2]
    test_year = years[-1]

    train = data[
        data["year"] < validation_year
    ].copy()

    validation = data[
        data["year"] == validation_year
    ].copy()

    test = data[
        data["year"] == test_year
    ].copy()

    for name, part in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if part.empty:
            raise ValueError(f"{name} split is empty.")

        if part[TARGET].nunique() < 2:
            raise ValueError(
                f"{name} split contains only one target class."
            )

    return train, validation, test


def get_numeric_features():
    company_counts = [
        f"company_{marker}_count"
        for marker in MARKERS
    ]
    company_shares = [
        f"company_{marker}_share"
        for marker in MARKERS
    ]

    return [
        *MARKERS,
        "current_marker_count",
        "multiple_current_markers",
        "years_since_first_observation",
        "prior_transition_count",
        "prior_change_count",
        "prior_change_rate",
        "previous_transition_changed",
        "has_previous_transition",
        "previous_transition_was_adjacent",
        "company_tmt_size",
        "company_role_count",
        "company_prior_changes",
        "company_prior_transitions",
        "company_prior_change_rate",
        *company_counts,
        *company_shares,
    ]


def prepare_numeric_data(
    train,
    validation,
    test,
    features,
):
    medians = train[features].median()

    train_x = train[features].fillna(medians)
    validation_x = validation[features].fillna(medians)
    test_x = test[features].fillna(medians)

    return train_x, validation_x, test_x


def probability_metrics(y_true, probability):
    return {
        "pr_auc": float(
            average_precision_score(y_true, probability)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
    }


def classification_metrics(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        **probability_metrics(y_true, probability),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, probability):
    rows = []

    for threshold in THRESHOLDS:
        rows.append(
            classification_metrics(
                y_true,
                probability,
                threshold,
            )
        )

    table = pd.DataFrame(rows)

    best = table.sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(best["threshold"]), table


def train_numeric_models(
    train,
    validation,
    train_x,
    validation_x,
):
    rows = []
    fitted = {}

    weights = compute_sample_weight(
        class_weight="balanced",
        y=train[TARGET],
    )

    for candidate_id, parameters in enumerate(
        GB_CANDIDATES,
        start=1,
    ):
        model = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **parameters,
        )

        model.fit(
            train_x,
            train[TARGET],
            sample_weight=weights,
        )

        probability = model.predict_proba(
            validation_x
        )[:, 1]

        metrics = probability_metrics(
            validation[TARGET],
            probability,
        )

        rows.append({
            "candidate_id": candidate_id,
            **parameters,
            **metrics,
        })

        fitted[candidate_id] = model

    tuning = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc"],
        ascending=False,
    )

    best_id = int(tuning.iloc[0]["candidate_id"])

    return (
        GB_CANDIDATES[best_id - 1],
        tuning,
    )


def train_text_model(train):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=20000,
        sublinear_tf=True,
    )

    train_x = vectorizer.fit_transform(
        train["role_text"]
    )

    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(train_x, train[TARGET])

    return vectorizer, model


def choose_ensemble(
    y_true,
    numeric_probability,
    text_probability,
):
    rows = []

    for numeric_weight in ENSEMBLE_WEIGHTS:
        probability = (
            numeric_weight * numeric_probability
            + (1.0 - numeric_weight) * text_probability
        )

        threshold, _ = select_threshold(
            y_true,
            probability,
        )

        metrics = classification_metrics(
            y_true,
            probability,
            threshold,
        )

        rows.append({
            "numeric_weight": float(numeric_weight),
            "text_weight": float(1.0 - numeric_weight),
            **metrics,
        })

    tuning = pd.DataFrame(rows).sort_values(
        ["pr_auc", "f1", "roc_auc"],
        ascending=False,
    )

    best = tuning.iloc[0]

    return (
        float(best["numeric_weight"]),
        float(best["threshold"]),
        tuning,
    )


def main():
    source, transitions = load_data()
    data = build_features(source, transitions)

    train, validation, test = split_by_time(data)

    numeric_features = get_numeric_features()

    (
        train_numeric,
        validation_numeric,
        test_numeric,
    ) = prepare_numeric_data(
        train,
        validation,
        test,
        numeric_features,
    )

    (
        best_numeric_parameters,
        numeric_tuning,
    ) = train_numeric_models(
        train,
        validation,
        train_numeric,
        validation_numeric,
    )

    numeric_model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **best_numeric_parameters,
    )

    train_weights = compute_sample_weight(
        class_weight="balanced",
        y=train[TARGET],
    )

    numeric_model.fit(
        train_numeric,
        train[TARGET],
        sample_weight=train_weights,
    )

    validation_numeric_probability = (
        numeric_model.predict_proba(
            validation_numeric
        )[:, 1]
    )

    vectorizer, text_model = train_text_model(train)

    validation_text_matrix = vectorizer.transform(
        validation["role_text"]
    )

    validation_text_probability = (
        text_model.predict_proba(
            validation_text_matrix
        )[:, 1]
    )

    (
        numeric_weight,
        ensemble_threshold,
        ensemble_tuning,
    ) = choose_ensemble(
        validation[TARGET],
        validation_numeric_probability,
        validation_text_probability,
    )

    # Refit models on train + validation only after selection.
    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    medians = development[numeric_features].median()
    development_numeric = (
        development[numeric_features].fillna(medians)
    )
    test_numeric = test[numeric_features].fillna(medians)

    final_numeric_model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **best_numeric_parameters,
    )

    development_weights = compute_sample_weight(
        class_weight="balanced",
        y=development[TARGET],
    )

    final_numeric_model.fit(
        development_numeric,
        development[TARGET],
        sample_weight=development_weights,
    )

    final_vectorizer, final_text_model = train_text_model(
        development
    )

    test_numeric_probability = (
        final_numeric_model.predict_proba(
            test_numeric
        )[:, 1]
    )

    test_text_probability = (
        final_text_model.predict_proba(
            final_vectorizer.transform(test["role_text"])
        )[:, 1]
    )

    test_ensemble_probability = (
        numeric_weight * test_numeric_probability
        + (1.0 - numeric_weight) * test_text_probability
    )

    numeric_threshold, _ = select_threshold(
        validation[TARGET],
        validation_numeric_probability,
    )

    text_threshold, _ = select_threshold(
        validation[TARGET],
        validation_text_probability,
    )

    prevalence = float(development[TARGET].mean())
    baseline_probability = np.full(
        len(test),
        prevalence,
    )

    evaluations = pd.DataFrame([
        {
            "model": "numeric_gradient_boosting",
            **classification_metrics(
                test[TARGET],
                test_numeric_probability,
                numeric_threshold,
            ),
        },
        {
            "model": "role_text_logistic_regression",
            **classification_metrics(
                test[TARGET],
                test_text_probability,
                text_threshold,
            ),
        },
        {
            "model": "numeric_text_ensemble",
            **classification_metrics(
                test[TARGET],
                test_ensemble_probability,
                ensemble_threshold,
            ),
        },
        {
            "model": "always_no_change_baseline",
            **classification_metrics(
                test[TARGET],
                baseline_probability,
                1.0,
            ),
        },
    ])

    for column, value in {
        "train_max_year": int(train["year"].max()),
        "validation_year": int(validation["year"].iloc[0]),
        "test_year": int(test["year"].iloc[0]),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "test_positive_share": float(test[TARGET].mean()),
    }.items():
        evaluations[column] = value

    predictions = test[
        [
            "person_key",
            "company",
            "GV_KEY",
            "full_name",
            "role",
            "year",
            "next_year",
            TARGET,
        ]
    ].copy()

    predictions["numeric_probability"] = (
        test_numeric_probability
    )
    predictions["text_probability"] = (
        test_text_probability
    )
    predictions["ensemble_probability"] = (
        test_ensemble_probability
    )
    predictions["ensemble_threshold"] = (
        ensemble_threshold
    )
    predictions["ensemble_prediction"] = (
        test_ensemble_probability
        >= ensemble_threshold
    ).astype(int)

    numeric_importance = pd.DataFrame({
        "feature": numeric_features,
        "importance": final_numeric_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    terms = final_vectorizer.get_feature_names_out()
    coefficients = final_text_model.coef_[0]

    text_importance = pd.DataFrame({
        "term": terms,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    }).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    ensemble_selection = pd.DataFrame([{
        "numeric_weight": numeric_weight,
        "text_weight": 1.0 - numeric_weight,
        "selected_threshold": ensemble_threshold,
    }])

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:
        evaluations.to_excel(
            writer,
            sheet_name="test_metrics",
            index=False,
        )
        predictions.to_excel(
            writer,
            sheet_name="test_predictions",
            index=False,
        )
        numeric_tuning.to_excel(
            writer,
            sheet_name="numeric_tuning",
            index=False,
        )
        ensemble_tuning.to_excel(
            writer,
            sheet_name="ensemble_tuning",
            index=False,
        )
        ensemble_selection.to_excel(
            writer,
            sheet_name="selected_ensemble",
            index=False,
        )
        numeric_importance.to_excel(
            writer,
            sheet_name="numeric_importance",
            index=False,
        )
        text_importance.head(500).to_excel(
            writer,
            sheet_name="text_importance",
            index=False,
        )

    print(
        f"Enriched Step 3 result saved to: "
        f"{OUTPUT_FILE.resolve()}"
    )

    print("\nSELECTED NUMERIC PARAMETERS")
    print(best_numeric_parameters)

    print("\nSELECTED ENSEMBLE")
    print(
        f"Numeric weight: {numeric_weight:.2f}\n"
        f"Text weight:    {1.0 - numeric_weight:.2f}\n"
        f"Threshold:      {ensemble_threshold:.2f}"
    )

    print("\nTEST RESULTS")
    print(
        evaluations[
            [
                "model",
                "pr_auc",
                "roc_auc",
                "f1",
                "precision",
                "recall",
                "balanced_accuracy",
                "accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()