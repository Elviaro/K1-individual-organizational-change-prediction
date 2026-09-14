"""Alternative Step 3: rolling-year evaluation.

For every test year:
- training years are earlier than the validation year;
- validation is the year immediately before the test year;
- test is one completely unseen year;
- parameters and classification threshold are selected without test data.

Default test years: 2019, 2020, 2021, 2022, 2023.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter

from sklearn.ensemble import GradientBoostingClassifier
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


INPUT_FILE = Path("person_step_2_transitions.xlsx")
OUTPUT_FILE = Path("person_step_3_rolling_result.xlsx")
INPUT_SHEET = "person_transitions"

TARGET = "y_marker_set_changed"
TEST_YEARS = [2019, 2020, 2021, 2022, 2023]
RANDOM_STATE = 42

MARKERS = [
    "vp", "svp", "evp", "sevp", "dir", "sdir",
    "md", "smd", "se", "vc", "svc", "president",
    "board", "ceo", "cxo", "primary", "support", "bu",
]

CANDIDATES = [
    {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_samples_leaf": 20,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 20,
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

THRESHOLDS = np.arange(0.05, 0.951, 0.01)

def progress(message):
    """Immediately display the current calculation stage."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Step 2 result not found: {path.resolve()}"
        )

    data = pd.read_excel(
        path,
        sheet_name=INPUT_SHEET,
    )

    required = {
        "person_key",
        "company",
        "GV_KEY",
        "full_name",
        "year",
        "next_year",
        TARGET,
        *MARKERS,
    }

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data["year"] = pd.to_numeric(
        data["year"], errors="raise"
    ).astype(int)

    data["next_year"] = pd.to_numeric(
        data["next_year"], errors="raise"
    ).astype(int)

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(
            f"{TARGET} must contain only 0 and 1."
        )

    for marker in MARKERS:
        if data[marker].isna().any():
            raise ValueError(
                f"{marker} contains missing values."
            )

        if not data[marker].isin([0, 1]).all():
            raise ValueError(
                f"{marker} contains non-binary values."
            )

    return data.sort_values(
        ["year", "person_key"]
    ).reset_index(drop=True)


def probability_metrics(y_true, probability):
    if pd.Series(y_true).nunique() < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(
            roc_auc_score(y_true, probability)
        )

    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "roc_auc": roc_auc,
    }


def complete_metrics(
    y_true,
    probability,
    prediction,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        **probability_metrics(y_true, probability),
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
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


def fit_model(data, parameters):
    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **parameters,
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=data[TARGET],
    )

    model.fit(
        data[MARKERS],
        data[TARGET],
        sample_weight=weights,
    )

    return model


def select_threshold(
    y_true,
    probability,
):
    rows = []

    for threshold in THRESHOLDS:
        prediction = (
            probability >= threshold
        ).astype(int)

        metrics = complete_metrics(
            y_true,
            probability,
            prediction,
        )

        rows.append({
            "threshold": float(threshold),
            **metrics,
        })

    table = pd.DataFrame(rows)

    # F1 determines the threshold.
    # Precision resolves exact ties.
    best = table.sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(best["threshold"]), table


def select_model(
    train,
    validation,
    test_year,
):
    rows = []
    probabilities = {}

    total_candidates = len(CANDIDATES)

    for candidate_id, parameters in enumerate(
        CANDIDATES,
        start=1,
    ):
        started = perf_counter()

        progress(
            f"Тестовый год {test_year}: "
            f"обучение модели {candidate_id}/{total_candidates}..."
        )

        model = fit_model(
            train,
            parameters,
        )

        progress(
            f"Тестовый год {test_year}: "
            f"модель {candidate_id}/{total_candidates} обучена, "
            "рассчитываем validation..."
        )

        probability = model.predict_proba(
            validation[MARKERS]
        )[:, 1]

        threshold, _ = select_threshold(
            validation[TARGET],
            probability,
        )

        prediction = (
            probability >= threshold
        ).astype(int)

        metrics = complete_metrics(
            validation[TARGET],
            probability,
            prediction,
        )

        rows.append({
            "test_year": test_year,
            "candidate_id": candidate_id,
            **parameters,
            "selected_threshold": threshold,
            **metrics,
        })

        probabilities[candidate_id] = probability

        elapsed = perf_counter() - started

        progress(
            f"Тестовый год {test_year}: "
            f"модель {candidate_id}/{total_candidates} готова — "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"{elapsed:.1f} сек."
        )

    progress(
        f"Тестовый год {test_year}: "
        "выбираем лучшую модель..."
    )

    tuning = pd.DataFrame(rows).sort_values(
        ["pr_auc", "f1", "roc_auc"],
        ascending=[False, False, False],
    )

    best = tuning.iloc[0]
    best_id = int(best["candidate_id"])
    best_parameters = CANDIDATES[best_id - 1]

    _, threshold_table = select_threshold(
        validation[TARGET],
        probabilities[best_id],
    )

    threshold_table.insert(
        0,
        "test_year",
        test_year,
    )
    threshold_table.insert(
        1,
        "candidate_id",
        best_id,
    )

    progress(
        f"Тестовый год {test_year}: выбрана модель {best_id}, "
        f"порог={float(best['selected_threshold']):.2f}."
    )

    return (
        best_parameters,
        float(best["selected_threshold"]),
        tuning,
        threshold_table,
    )

def build_fold(
    data,
    test_year,
):
    validation_year = test_year - 1

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
            raise ValueError(
                f"{name} is empty for test year "
                f"{test_year}."
            )

        if part[TARGET].nunique() < 2:
            raise ValueError(
                f"{name} has only one target class "
                f"for test year {test_year}."
            )

    return train, validation, test


def run_fold(
    data,
    test_year,
):
    fold_started = perf_counter()

    progress(
        f"Тестовый год {test_year}: формируем временные выборки..."
    )

    train, validation, test = build_fold(
        data,
        test_year,
    )

    progress(
        f"Тестовый год {test_year}: выборки готовы — "
        f"train={len(train):,}, "
        f"validation={len(validation):,}, "
        f"test={len(test):,}."
    )

    (
        parameters,
        threshold,
        tuning,
        threshold_table,
    ) = select_model(
        train,
        validation,
        test_year,
    )

    progress(
        f"Тестовый год {test_year}: "
        "переобучаем выбранную модель на train + validation..."
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    model = fit_model(
        development,
        parameters,
    )

    progress(
        f"Тестовый год {test_year}: "
        "строим прогноз для скрытого тестового года..."
    )

    probability = model.predict_proba(
        test[MARKERS]
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    baseline_probability = np.full(
        len(test),
        float(development[TARGET].mean()),
    )
    baseline_prediction = np.zeros(
        len(test),
        dtype=int,
    )

    common = {
        "test_year": test_year,
        "validation_year": test_year - 1,
        "train_min_year": int(train["year"].min()),
        "train_max_year": int(train["year"].max()),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "test_positive_share": float(
            test[TARGET].mean()
        ),
    }

    model_metrics = complete_metrics(
        test[TARGET],
        probability,
        prediction,
    )

    baseline_metrics = complete_metrics(
        test[TARGET],
        baseline_probability,
        baseline_prediction,
    )

    model_row = {
        "model": "gradient_boosting_classifier",
        **common,
        **parameters,
        "threshold": threshold,
        **model_metrics,
    }

    baseline_row = {
        "model": "always_no_change_baseline",
        **common,
        "n_estimators": None,
        "learning_rate": None,
        "max_depth": None,
        "min_samples_leaf": None,
        "threshold": 1.0,
        **baseline_metrics,
    }

    predictions = test[
        [
            "person_key",
            "company",
            "GV_KEY",
            "full_name",
            "year",
            "next_year",
            TARGET,
        ]
    ].copy()

    predictions["test_year"] = test_year
    predictions["predicted_probability"] = probability
    predictions["selected_threshold"] = threshold
    predictions["predicted_class"] = prediction
    predictions["baseline_probability"] = (
        baseline_probability
    )
    predictions["baseline_class"] = 0

    importance = pd.DataFrame({
        "test_year": test_year,
        "feature": MARKERS,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    elapsed = perf_counter() - fold_started

    progress(
        f"Тестовый год {test_year}: ГОТОВО — "
        f"PR-AUC={model_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={model_metrics['roc_auc']:.4f}, "
        f"F1={model_metrics['f1']:.4f}; "
        f"время этапа {elapsed:.1f} сек."
    )

    return {
        "metrics": [model_row, baseline_row],
        "predictions": predictions,
        "tuning": tuning,
        "thresholds": threshold_table,
        "importance": importance,
    }


def pooled_evaluation(predictions):
    y_true = predictions[TARGET]
    model_probability = predictions[
        "predicted_probability"
    ].to_numpy()
    model_prediction = predictions[
        "predicted_class"
    ].to_numpy()

    baseline_probability = predictions[
        "baseline_probability"
    ].to_numpy()
    baseline_prediction = predictions[
        "baseline_class"
    ].to_numpy()

    return pd.DataFrame([
        {
            "model": "gradient_boosting_classifier",
            "evaluation": "pooled_rolling_years",
            "n_test": len(predictions),
            "n_positive": int(y_true.sum()),
            "positive_share": float(y_true.mean()),
            **complete_metrics(
                y_true,
                model_probability,
                model_prediction,
            ),
        },
        {
            "model": "always_no_change_baseline",
            "evaluation": "pooled_rolling_years",
            "n_test": len(predictions),
            "n_positive": int(y_true.sum()),
            "positive_share": float(y_true.mean()),
            **complete_metrics(
                y_true,
                baseline_probability,
                baseline_prediction,
            ),
        },
    ])


def yearly_summary(yearly_metrics):
    model = yearly_metrics[
        yearly_metrics["model"]
        == "gradient_boosting_classifier"
    ]

    columns = [
        "pr_auc",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "accuracy",
    ]

    rows = []

    for column in columns:
        rows.append({
            "metric": column,
            "mean_across_years": float(
                model[column].mean()
            ),
            "std_across_years": float(
                model[column].std()
            ),
            "minimum": float(
                model[column].min()
            ),
            "maximum": float(
                model[column].max()
            ),
        })

    return pd.DataFrame(rows)


def main():
    total_started = perf_counter()

    progress("ШАГ 1/7: загружаем person_step_2_transitions.xlsx...")
    data = load_data(INPUT_FILE)

    progress(
        f"ШАГ 1/7 завершён: загружено "
        f"{len(data):,} переходов."
    )

    progress("ШАГ 2/7: проверяем доступность тестовых лет...")

    available_years = set(data["year"].unique())
    missing_test_years = [
        year
        for year in TEST_YEARS
        if year not in available_years
        or year - 1 not in available_years
    ]

    if missing_test_years:
        raise ValueError(
            "Unavailable test/validation years: "
            + ", ".join(map(str, missing_test_years))
        )

    progress(
        "ШАГ 2/7 завершён: будут проверены годы "
        + ", ".join(map(str, TEST_YEARS))
        + "."
    )

    all_metrics = []
    all_predictions = []
    all_tuning = []
    all_thresholds = []
    all_importance = []

    progress("ШАГ 3/7: запускаем последовательные годовые тесты...")

    for fold_number, test_year in enumerate(
        TEST_YEARS,
        start=1,
    ):
        progress(
            f"ГОДОВОЙ ТЕСТ {fold_number}/{len(TEST_YEARS)}: "
            f"начинаем прогноз для {test_year} года."
        )

        result = run_fold(
            data,
            test_year,
        )

        all_metrics.extend(result["metrics"])
        all_predictions.append(result["predictions"])
        all_tuning.append(result["tuning"])
        all_thresholds.append(result["thresholds"])
        all_importance.append(result["importance"])

        progress(
            f"ГОДОВОЙ ТЕСТ {fold_number}/{len(TEST_YEARS)} завершён."
        )

    progress("ШАГ 3/7 завершён: все годовые тесты рассчитаны.")

    progress("ШАГ 4/7: объединяем результаты всех лет...")

    yearly_metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )
    tuning = pd.concat(
        all_tuning,
        ignore_index=True,
    )
    thresholds = pd.concat(
        all_thresholds,
        ignore_index=True,
    )
    importance = pd.concat(
        all_importance,
        ignore_index=True,
    )

    progress("ШАГ 4/7 завершён.")

    progress("ШАГ 5/7: рассчитываем общие метрики...")

    pooled = pooled_evaluation(predictions)
    stability = yearly_summary(yearly_metrics)

    progress("ШАГ 5/7 завершён.")

    progress(
        f"ШАГ 6/7: сохраняем результаты в {OUTPUT_FILE.name}..."
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:
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
        predictions.to_excel(
            writer,
            sheet_name="all_test_predictions",
            index=False,
        )
        tuning.to_excel(
            writer,
            sheet_name="validation_tuning",
            index=False,
        )
        thresholds.to_excel(
            writer,
            sheet_name="threshold_tuning",
            index=False,
        )
        importance.to_excel(
            writer,
            sheet_name="feature_importance",
            index=False,
        )

    progress(
        f"ШАГ 6/7 завершён: файл сохранён — "
        f"{OUTPUT_FILE.resolve()}"
    )

    progress("ШАГ 7/7: выводим итоговые таблицы...")

    display_columns = [
        "model",
        "test_year",
        "n_test",
        "test_positive_share",
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

    print("\nYEARLY TEST RESULTS", flush=True)
    print(
        yearly_metrics[display_columns]
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    print("\nPOOLED OUT-OF-SAMPLE RESULTS", flush=True)
    print(
        pooled[
            [
                "model",
                "n_test",
                "positive_share",
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
        .to_string(index=False),
        flush=True,
    )

    print("\nSTABILITY ACROSS YEARS", flush=True)
    print(
        stability
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    total_elapsed = perf_counter() - total_started

    progress(
        f"ШАГ 7/7 завершён. ВСЯ ПРОГРАММА ГОТОВА. "
        f"Общее время: {total_elapsed / 60:.1f} мин."
    )

if __name__ == "__main__":
    main()