"""Step 6: audit personnel disappearances and test turnover waves.

Place this file next to ``20260721_OD_Database_1993_2024b.xlsx`` and run:

    python person_step_6_turnover_wave_audit.py

The script distinguishes changes in structural markers from disappearance of a
person from the observed top-management-team roster.  A disappearance is not
called a dismissal: annual data only show that a person is no longer listed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


INPUT_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("person_step_6_turnover_wave_audit.xlsx")

MARKERS = [
    "vp", "svp", "evp", "sevp", "dir", "sdir", "md", "smd", "se",
    "vc", "svc", "president", "board", "ceo", "cxo", "primary",
    "support", "bu",
]

NAME_SIMILARITY_THRESHOLD = 0.92
COVERAGE_DROP_THRESHOLD = 0.50
WAVE_MIN_DISAPPEARANCES = 3
WAVE_TAIL_PROBABILITY = 0.05
BASELINE_SMOOTHING = 20.0


def progress(message: str) -> None:
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def normalize_words(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def source_set(values: pd.Series) -> str:
    sources: set[str] = set()
    for value in values.dropna().astype(str):
        sources.update(part.strip() for part in value.split("|") if part.strip())
    return "|".join(sorted(sources))


def binomial_tail(n: int, p: float, k: int) -> float:
    """P(X >= k) for X ~ Binomial(n, p), without a SciPy dependency."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(
        sum(
            math.comb(n, x) * (p ** x) * ((1.0 - p) ** (n - x))
            for x in range(k, n + 1)
        )
    )


def validate_columns(data: pd.DataFrame) -> None:
    required = {"year", "company", "GV_KEY", "full_name", "TMTSource", *MARKERS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def load_person_years() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}\n"
            "Place the script next to the Excel database."
        )

    progress(f"STEP 6.1/7: loading {INPUT_FILE.name}...")
    usecols = [
        "year", "company", "GV_KEY", "full_name", "TMTSource", *MARKERS,
    ]
    data = pd.read_excel(INPUT_FILE, sheet_name=0, usecols=usecols)
    validate_columns(data)

    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["GV_KEY"] = pd.to_numeric(data["GV_KEY"], errors="coerce")
    data["name_words"] = data["full_name"].map(normalize_words)
    data = data[
        data["year"].notna()
        & data["GV_KEY"].notna()
        & data["name_words"].ne("")
    ].copy()
    data["year"] = data["year"].astype(int)
    data["GV_KEY"] = data["GV_KEY"].astype("int64")
    data["name_key"] = data["name_words"].str.replace(" ", "", regex=False)
    data["person_key"] = (
        data["GV_KEY"].astype(str) + "|" + data["name_key"]
    )

    for marker in MARKERS:
        data[marker] = (
            pd.to_numeric(data[marker], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype("int8")
        )

    aggregation = {marker: "max" for marker in MARKERS}
    aggregation.update({
        "company": "first",
        "full_name": "first",
        "name_words": "first",
        "TMTSource": source_set,
    })
    person_years = (
        data.groupby(["GV_KEY", "year", "person_key"], as_index=False)
        .agg(aggregation)
        .sort_values(["GV_KEY", "year", "person_key"])
        .reset_index(drop=True)
    )
    progress(
        f"STEP 6.1/7 complete: {len(person_years):,} person-years, "
        f"{person_years['GV_KEY'].nunique():,} companies."
    )
    return person_years


def build_company_years(person_years: pd.DataFrame) -> pd.DataFrame:
    progress("STEP 6.2/7: building company-year coverage diagnostics...")
    company_years = (
        person_years.groupby(["GV_KEY", "year"], as_index=False)
        .agg(
            company=("company", "first"),
            team_size=("person_key", "size"),
            source=("TMTSource", source_set),
        )
        .sort_values(["GV_KEY", "year"])
        .reset_index(drop=True)
    )
    company_years["company_first_year"] = company_years.groupby("GV_KEY")[
        "year"
    ].transform("min")
    company_years["company_last_year"] = company_years.groupby("GV_KEY")[
        "year"
    ].transform("max")
    observed = set(zip(company_years["GV_KEY"], company_years["year"]))
    company_years["next_year_observed"] = [
        (company, year + 1) in observed
        for company, year in zip(company_years["GV_KEY"], company_years["year"])
    ]
    company_years["two_years_ahead_observed"] = [
        (company, year + 2) in observed
        for company, year in zip(company_years["GV_KEY"], company_years["year"])
    ]
    return company_years


def find_likely_renames(
    transition_people: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Match disappearing names to highly similar entering names."""
    likely = pd.Series(False, index=transition_people.index)
    matched_name = pd.Series("", index=transition_people.index, dtype="object")

    exits = transition_people[transition_people["one_year_disappearance"]]
    for (_, _), group in exits.groupby(["GV_KEY", "year"], sort=False):
        entry_names = group.attrs.get("entry_names", [])
        # attrs are not preserved by groupby; entry candidates are attached below.
        if not entry_names:
            continue
        for index, row in group.iterrows():
            best_name = ""
            best_score = 0.0
            for candidate in entry_names:
                score = SequenceMatcher(None, row["name_words"], candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_name = candidate
            if best_score >= NAME_SIMILARITY_THRESHOLD:
                likely.at[index] = True
                matched_name.at[index] = best_name
    return likely, matched_name


def add_name_match_flags(
    people: pd.DataFrame,
    person_years: pd.DataFrame,
) -> pd.DataFrame:
    """Vector-friendly wrapper for fuzzy matching within a company transition."""
    entry_map: dict[tuple[int, int], list[str]] = {}
    previous = person_years[["GV_KEY", "year", "person_key"]].copy()
    previous["year"] += 1
    arrivals = person_years.merge(
        previous.assign(previously_observed=True),
        on=["GV_KEY", "year", "person_key"],
        how="left",
    )
    arrivals = arrivals[arrivals["previously_observed"].isna()].copy()
    arrivals["origin_year"] = arrivals["year"] - 1
    for key, group in arrivals.groupby(["GV_KEY", "origin_year"], sort=False):
        entry_map[key] = group["name_words"].tolist()

    likely = np.zeros(len(people), dtype=bool)
    matched = np.full(len(people), "", dtype=object)
    candidates = people.index[people["one_year_disappearance"]]
    for counter, index in enumerate(candidates, start=1):
        row = people.loc[index]
        names = entry_map.get((int(row["GV_KEY"]), int(row["year"])), [])
        best_score = 0.0
        best_name = ""
        for candidate in names:
            score = SequenceMatcher(None, row["name_words"], candidate).ratio()
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_score >= NAME_SIMILARITY_THRESHOLD:
            likely[index] = True
            matched[index] = best_name
        if counter % 10000 == 0:
            progress(f"  checked {counter:,}/{len(candidates):,} disappearances...")

    people["likely_name_variant"] = likely
    people["matched_new_name"] = matched
    return people


def build_person_status(
    person_years: pd.DataFrame,
    company_years: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress("STEP 6.3/7: classifying stayers, disappearances and censoring...")
    observed_people = set(
        zip(
            person_years["GV_KEY"],
            person_years["year"],
            person_years["person_key"],
        )
    )
    observed_company_years = set(
        zip(company_years["GV_KEY"], company_years["year"])
    )

    people = person_years[
        ["GV_KEY", "company", "year", "person_key", "full_name", "name_words"]
    ].copy()
    people["company_next_year_observed"] = [
        (company, year + 1) in observed_company_years
        for company, year in zip(people["GV_KEY"], people["year"])
    ]
    people["company_two_years_ahead_observed"] = [
        (company, year + 2) in observed_company_years
        for company, year in zip(people["GV_KEY"], people["year"])
    ]
    people["person_next_year_observed"] = [
        (company, year + 1, person) in observed_people
        for company, year, person in zip(
            people["GV_KEY"], people["year"], people["person_key"]
        )
    ]
    people["person_two_years_ahead_observed"] = [
        (company, year + 2, person) in observed_people
        for company, year, person in zip(
            people["GV_KEY"], people["year"], people["person_key"]
        )
    ]
    people["right_censored"] = ~people["company_next_year_observed"]
    people["one_year_disappearance"] = (
        people["company_next_year_observed"]
        & ~people["person_next_year_observed"]
    )
    people["temporary_gap"] = (
        people["one_year_disappearance"]
        & people["person_two_years_ahead_observed"]
    )
    people["persistent_disappearance"] = (
        people["one_year_disappearance"]
        & people["company_two_years_ahead_observed"]
        & ~people["person_two_years_ahead_observed"]
    )
    people = add_name_match_flags(people, person_years)

    current = company_years[
        ["GV_KEY", "year", "team_size", "source", "company_last_year"]
    ].rename(columns={"team_size": "team_size_t", "source": "source_t"})
    following = company_years[["GV_KEY", "year", "team_size", "source"]].copy()
    following["year"] -= 1
    following = following.rename(
        columns={"team_size": "team_size_t1", "source": "source_t1"}
    )
    people = people.merge(current, on=["GV_KEY", "year"], how="left")
    people = people.merge(following, on=["GV_KEY", "year"], how="left")
    people["source_changed"] = (
        people["company_next_year_observed"]
        & people["source_t"].fillna("").ne(people["source_t1"].fillna(""))
    )
    people["coverage_drop"] = (
        people["company_next_year_observed"]
        & people["team_size_t1"].lt(
            people["team_size_t"] * COVERAGE_DROP_THRESHOLD
        )
    )
    people["comparable_coverage"] = (
        people["company_next_year_observed"]
        & ~people["source_changed"]
        & ~people["coverage_drop"]
    )

    conditions = [
        people["right_censored"],
        people["person_next_year_observed"],
        people["source_changed"] | people["coverage_drop"],
        people["likely_name_variant"],
        people["temporary_gap"],
        people["one_year_disappearance"],
    ]
    labels = [
        "right_censored",
        "observed_stayer",
        "coverage_ambiguous",
        "possible_name_variant",
        "temporary_gap",
        "observed_disappearance",
    ]
    people["status"] = np.select(conditions, labels, default="unknown")
    people["reliable_one_year_disappearance"] = (
        people["one_year_disappearance"]
        & people["comparable_coverage"]
        & ~people["likely_name_variant"]
        & ~people["temporary_gap"]
    )
    people["reliable_persistent_disappearance"] = (
        people["reliable_one_year_disappearance"]
        & people["persistent_disappearance"]
    )

    transition_people = people[people["company_next_year_observed"]].copy()
    return people, transition_people


def build_transition_table(
    transition_people: pd.DataFrame,
    person_years: pd.DataFrame,
) -> pd.DataFrame:
    progress("STEP 6.4/7: aggregating company-year turnover transitions...")
    grouped = (
        transition_people.groupby(["GV_KEY", "company", "year"], as_index=False)
        .agg(
            team_size_t=("team_size_t", "first"),
            team_size_t1=("team_size_t1", "first"),
            source_t=("source_t", "first"),
            source_t1=("source_t1", "first"),
            source_changed=("source_changed", "first"),
            coverage_drop=("coverage_drop", "first"),
            comparable_coverage=("comparable_coverage", "first"),
            two_years_ahead_observed=("company_two_years_ahead_observed", "first"),
            raw_disappearances=("one_year_disappearance", "sum"),
            temporary_gaps=("temporary_gap", "sum"),
            likely_name_variants=("likely_name_variant", "sum"),
            reliable_disappearances=("reliable_one_year_disappearance", "sum"),
            persistent_disappearances=("reliable_persistent_disappearance", "sum"),
        )
    )

    previous = person_years[["GV_KEY", "year", "person_key"]].copy()
    previous["year"] += 1
    arrivals = person_years.merge(
        previous.assign(previously_observed=True),
        on=["GV_KEY", "year", "person_key"],
        how="left",
    )
    arrivals["entry"] = arrivals["previously_observed"].isna()
    arrivals["year"] -= 1
    entries = (
        arrivals.groupby(["GV_KEY", "year"], as_index=False)["entry"]
        .sum()
        .rename(columns={"entry": "entries"})
    )
    grouped = grouped.merge(entries, on=["GV_KEY", "year"], how="left")
    grouped["entries"] = grouped["entries"].fillna(0).astype(int)
    grouped["raw_exit_rate"] = grouped["raw_disappearances"] / grouped["team_size_t"]
    grouped["reliable_exit_rate"] = (
        grouped["reliable_disappearances"] / grouped["team_size_t"]
    )
    grouped["persistent_exit_rate"] = (
        grouped["persistent_disappearances"] / grouped["team_size_t"]
    )
    grouped.loc[
        ~grouped["two_years_ahead_observed"],
        ["persistent_disappearances", "persistent_exit_rate"],
    ] = np.nan
    grouped["team_size_change"] = grouped["team_size_t1"] - grouped["team_size_t"]
    grouped["relative_team_size_change"] = (
        grouped["team_size_change"].abs() / grouped["team_size_t"].clip(lower=1)
    )
    return grouped.sort_values(["GV_KEY", "year"]).reset_index(drop=True)


def add_wave_statistics(transitions: pd.DataFrame) -> pd.DataFrame:
    progress("STEP 6.5/7: testing whether disappearances cluster into waves...")
    comparable = transitions[transitions["comparable_coverage"]].copy()
    global_exits = comparable["reliable_disappearances"].sum()
    global_risk = comparable["team_size_t"].sum()
    global_rate = global_exits / global_risk

    company_rates = (
        comparable.groupby("GV_KEY", as_index=False)
        .agg(company_exits=("reliable_disappearances", "sum"), company_risk=("team_size_t", "sum"))
    )
    transitions = transitions.merge(
        company_rates,
        on="GV_KEY",
        how="left",
    )
    # Leave the focal transition out of its own baseline.  Otherwise an
    # extreme year mechanically raises the probability used to judge it.
    comparable_k = transitions["reliable_disappearances"].where(
        transitions["comparable_coverage"], 0
    )
    comparable_n = transitions["team_size_t"].where(
        transitions["comparable_coverage"], 0
    )
    transitions["baseline_exit_probability"] = (
        transitions["company_exits"].fillna(0)
        - comparable_k
        + BASELINE_SMOOTHING * global_rate
    ) / (
        transitions["company_risk"].fillna(0)
        - comparable_n
        + BASELINE_SMOOTHING
    )
    transitions["wave_tail_probability"] = [
        binomial_tail(int(n), float(p), int(k))
        for n, p, k in zip(
            transitions["team_size_t"],
            transitions["baseline_exit_probability"],
            transitions["reliable_disappearances"],
        )
    ]
    transitions["wave_flag"] = (
        transitions["comparable_coverage"]
        & transitions["reliable_disappearances"].ge(WAVE_MIN_DISAPPEARANCES)
        & transitions["wave_tail_probability"].lt(WAVE_TAIL_PROBABILITY)
    )
    return transitions


def make_null_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = transitions[transitions["comparable_coverage"]].copy()
    for sample_name, event_column in [
        ("comparable_one_year", "reliable_disappearances"),
        ("comparable_persistent", "persistent_disappearances"),
    ]:
        selected = sample[sample[event_column].notna()].copy()
        total_events = selected[event_column].sum()
        total_risk = selected["team_size_t"].sum()
        global_rate = total_events / total_risk
        totals = selected.groupby("GV_KEY").agg(
            null_company_events=(event_column, "sum"),
            null_company_risk=("team_size_t", "sum"),
        )
        selected = selected.join(totals, on="GV_KEY")
        selected["null_probability"] = (
            selected["null_company_events"]
            - selected[event_column]
            + BASELINE_SMOOTHING * global_rate
        ) / (
            selected["null_company_risk"]
            - selected["team_size_t"]
            + BASELINE_SMOOTHING
        )
        for threshold in [2, 3, 4, 5]:
            observed = int(selected[event_column].ge(threshold).sum())
            expected = float(
                sum(
                    binomial_tail(int(n), float(p), threshold)
                    for n, p in zip(
                        selected["team_size_t"],
                        selected["null_probability"],
                    )
                )
            )
            rows.append({
                "sample": sample_name,
                "threshold": f">={threshold} disappearances",
                "n_company_year_transitions": len(selected),
                "observed_transitions": observed,
                "expected_under_independence": expected,
                "observed_to_expected_ratio": observed / expected if expected else np.nan,
            })
    return pd.DataFrame(rows)


def make_cascade_table(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = transitions.copy()
    following = transitions[
        [
            "GV_KEY", "year", "reliable_disappearances", "reliable_exit_rate",
            "wave_flag", "comparable_coverage",
        ]
    ].copy()
    following["year"] -= 1
    following = following.rename(columns={
        "reliable_disappearances": "next_reliable_disappearances",
        "reliable_exit_rate": "next_reliable_exit_rate",
        "wave_flag": "next_wave_flag",
        "comparable_coverage": "next_comparable_coverage",
    })
    cascade = current.merge(following, on=["GV_KEY", "year"], how="inner")
    cascade = cascade[
        cascade["comparable_coverage"]
        & cascade["next_comparable_coverage"]
        & cascade["next_reliable_exit_rate"].notna()
    ].copy()
    cascade["current_exit_group"] = pd.cut(
        cascade["reliable_disappearances"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=["0", "1", "2", "3+"],
    )
    dose = (
        cascade.groupby("current_exit_group", observed=True, as_index=False)
        .agg(
            n_transitions=("GV_KEY", "size"),
            current_mean_exit_rate=("reliable_exit_rate", "mean"),
            next_mean_disappearances=("next_reliable_disappearances", "mean"),
            next_mean_exit_rate=("next_reliable_exit_rate", "mean"),
            next_wave_probability=("next_wave_flag", "mean"),
        )
    )
    zero_rate = dose.loc[
        dose["current_exit_group"].astype(str).eq("0"), "next_mean_exit_rate"
    ]
    baseline = float(zero_rate.iloc[0]) if len(zero_rate) else np.nan
    dose["next_exit_rate_difference_vs_zero"] = dose["next_mean_exit_rate"] - baseline
    dose["next_exit_rate_ratio_vs_zero"] = dose["next_mean_exit_rate"] / baseline
    return cascade, dose


def make_summaries(
    people: pd.DataFrame,
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = pd.DataFrame([
        {"metric": "person_years", "value": len(people)},
        {"metric": "right_censored_person_years", "value": int(people["right_censored"].sum())},
        {"metric": "observed_stayers", "value": int(people["person_next_year_observed"].sum())},
        {"metric": "raw_one_year_disappearances", "value": int(people["one_year_disappearance"].sum())},
        {"metric": "coverage_ambiguous_disappearances", "value": int((people["one_year_disappearance"] & ~people["comparable_coverage"]).sum())},
        {"metric": "likely_name_variants", "value": int(people["likely_name_variant"].sum())},
        {"metric": "temporary_gaps", "value": int(people["temporary_gap"].sum())},
        {"metric": "reliable_one_year_disappearances", "value": int(people["reliable_one_year_disappearance"].sum())},
        {"metric": "reliable_persistent_disappearances", "value": int(people["reliable_persistent_disappearance"].sum())},
        {"metric": "company_year_wave_flags", "value": int(transitions["wave_flag"].sum())},
    ])

    yearly = (
        transitions.groupby("year", as_index=False)
        .agg(
            n_companies=("GV_KEY", "size"),
            people_at_risk=("team_size_t", "sum"),
            raw_disappearances=("raw_disappearances", "sum"),
            reliable_disappearances=("reliable_disappearances", "sum"),
            persistent_disappearances=("persistent_disappearances", "sum"),
            entries=("entries", "sum"),
            source_changes=("source_changed", "sum"),
            coverage_drops=("coverage_drop", "sum"),
            waves=("wave_flag", "sum"),
        )
    )
    persistent_risk = (
        transitions[transitions["two_years_ahead_observed"]]
        .groupby("year")["team_size_t"]
        .sum()
        .rename("persistent_people_at_risk")
    )
    yearly = yearly.join(persistent_risk, on="year")
    yearly["raw_exit_rate"] = yearly["raw_disappearances"] / yearly["people_at_risk"]
    yearly["reliable_exit_rate"] = yearly["reliable_disappearances"] / yearly["people_at_risk"]
    yearly["persistent_exit_rate"] = (
        yearly["persistent_disappearances"] / yearly["persistent_people_at_risk"]
    )
    return quality, yearly


def save_workbook(
    people: pd.DataFrame,
    transitions: pd.DataFrame,
    null_summary: pd.DataFrame,
    cascade: pd.DataFrame,
    dose: pd.DataFrame,
    quality: pd.DataFrame,
    yearly: pd.DataFrame,
) -> None:
    progress(f"STEP 6.7/7: saving {OUTPUT_FILE.name}...")
    readme = pd.DataFrame({
        "item": [
            "Purpose",
            "Observed disappearance",
            "Reliable one-year disappearance",
            "Persistent disappearance",
            "Right censoring",
            "Wave",
            "Causal limitation",
        ],
        "definition": [
            "Audit clustering and temporal continuation of observed TMT roster disappearances.",
            "Person is listed in year t, company is observed in t+1, person is not listed in t+1.",
            "Observed disappearance without source change, major coverage drop, likely name variant or return in t+2.",
            "Reliable disappearance where company is observable in t+2 and the person remains absent.",
            "No adjacent company observation exists; exit status is unknown and excluded from exit rates.",
            "At least three reliable disappearances with binomial upper-tail probability below 0.05.",
            "Annual observational data identify clustering and temporal association, not interpersonal causation.",
        ],
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="readme", index=False)
        quality.to_excel(writer, sheet_name="quality_summary", index=False)
        yearly.to_excel(writer, sheet_name="yearly_summary", index=False)
        null_summary.to_excel(writer, sheet_name="wave_null_test", index=False)
        dose.to_excel(writer, sheet_name="cascade_dose", index=False)
        transitions.to_excel(writer, sheet_name="company_transitions", index=False)
        transitions[transitions["wave_flag"]].to_excel(
            writer, sheet_name="detected_waves", index=False
        )
        cascade.to_excel(writer, sheet_name="cascade_pairs", index=False)
        people[people["one_year_disappearance"]].to_excel(
            writer, sheet_name="disappearance_events", index=False
        )
        people.to_excel(writer, sheet_name="all_person_status", index=False)


def main() -> None:
    started = perf_counter()
    person_years = load_person_years()
    company_years = build_company_years(person_years)
    people, transition_people = build_person_status(person_years, company_years)
    transitions = build_transition_table(transition_people, person_years)
    transitions = add_wave_statistics(transitions)

    progress("STEP 6.6/7: evaluating independence and next-year continuation...")
    null_summary = make_null_summary(transitions)
    cascade, dose = make_cascade_table(transitions)
    quality, yearly = make_summaries(people, transitions)
    save_workbook(
        people, transitions, null_summary, cascade, dose, quality, yearly
    )

    print("\nQUALITY SUMMARY", flush=True)
    print(quality.to_string(index=False), flush=True)
    print("\nWAVE TEST: OBSERVED VS INDEPENDENT BASELINE", flush=True)
    print(null_summary.round(3).to_string(index=False), flush=True)
    print("\nNEXT-YEAR CASCADE DOSE RESPONSE", flush=True)
    print(dose.round(4).to_string(index=False), flush=True)
    print("\nRECENT YEARLY RESULTS", flush=True)
    print(yearly.tail(10).round(4).to_string(index=False), flush=True)
    progress(
        f"STEP 6 COMPLETE. Saved to {OUTPUT_FILE.resolve()}. "
        f"Total time: {(perf_counter() - started) / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
