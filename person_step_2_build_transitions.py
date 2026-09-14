"""Step 2: build individual adjacent-year marker transitions."""
from pathlib import Path
import re
import unicodedata
import pandas as pd

SOURCE_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
AUDIT_FILE = Path("person_step_1_linkage_audit.xlsx")
OUTPUT_FILE = Path("person_step_2_transitions.xlsx")
MARKERS = ["vp","svp","evp","sevp","dir","sdir","md","smd","se","vc","svc","president","board","ceo","cxo","primary","support","bu"]

def norm(value):
    if pd.isna(value): return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", value)

def main():
    for path in [SOURCE_FILE, AUDIT_FILE]:
        if not path.exists(): raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    pairs = pd.read_excel(AUDIT_FILE, sheet_name="adjacent_pairs")
    source = pd.read_excel(SOURCE_FILE)
    required = {"GV_KEY", "full_name", "year", *MARKERS}
    missing = sorted(required - set(source.columns))
    if missing: raise ValueError("Missing source columns: " + ", ".join(missing))

    source["org_key"] = source["GV_KEY"].astype("string").str.strip()
    source["name_key"] = source["full_name"].map(norm)
    source["person_key"] = source["org_key"].fillna("") + "|" + source["name_key"]
    states = source[["person_key", "year", *MARKERS]].copy()

    # Keep only globally unambiguous person-year states. Ambiguous keys were
    # excluded by Step 1 and must not block validation of unrelated pairs.
    unique_mask = ~states.duplicated(["person_key", "year"], keep=False)
    states = states.loc[unique_mask].copy()

    current = pairs[["person_key", "year", "next_year", "company", "GV_KEY", "full_name"]].merge(
        states, on=["person_key", "year"], how="left", validate="one_to_one"
    )
    next_states = states.rename(columns={"year": "next_year", **{c: f"next_{c}" for c in MARKERS}})
    transitions = current.merge(next_states, on=["person_key", "next_year"], how="left", validate="one_to_one")

    all_marker_columns = [*MARKERS, *[f"next_{c}" for c in MARKERS]]
    transitions["complete_marker_pair"] = transitions[all_marker_columns].notna().all(axis=1)
    incomplete = transitions.loc[~transitions["complete_marker_pair"]].copy()
    transitions = transitions.loc[transitions["complete_marker_pair"]].copy()
    if transitions.empty: raise ValueError("No complete marker pairs remain after validation.")

    delta_columns = []
    for c in MARKERS:
        dc = f"delta_{c}"; delta_columns.append(dc)
        transitions[dc] = transitions[f"next_{c}"] - transitions[c]
    transitions["changed_markers"] = transitions.apply(
        lambda row: "; ".join(c for c in MARKERS if row[f"delta_{c}"] != 0), axis=1
    )
    transitions["n_markers_changed"] = transitions[delta_columns].ne(0).sum(axis=1)
    transitions["y_marker_set_changed"] = transitions["n_markers_changed"].gt(0).astype(int)

    target_summary = pd.DataFrame({
        "class": ["no_marker_change", "marker_set_changed"],
        "n_transitions": [int(transitions["y_marker_set_changed"].eq(0).sum()), int(transitions["y_marker_set_changed"].eq(1).sum())],
    })
    target_summary["share_percent"] = target_summary["n_transitions"] / len(transitions) * 100
    marker_summary = pd.DataFrame([{
        "marker": c, "n_added": int(transitions[f"delta_{c}"].gt(0).sum()),
        "n_removed": int(transitions[f"delta_{c}"].lt(0).sum()),
        "n_unchanged": int(transitions[f"delta_{c}"].eq(0).sum()),
    } for c in MARKERS])
    processing_summary = pd.DataFrame({
        "metric": ["audited_adjacent_pairs", "complete_marker_pairs", "excluded_incomplete_marker_pairs"],
        "value": [len(pairs), len(transitions), len(incomplete)],
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        transitions.to_excel(writer, sheet_name="person_transitions", index=False)
        target_summary.to_excel(writer, sheet_name="target_summary", index=False)
        marker_summary.to_excel(writer, sheet_name="marker_change_summary", index=False)
        processing_summary.to_excel(writer, sheet_name="processing_summary", index=False)
        incomplete.to_excel(writer, sheet_name="excluded_incomplete", index=False)
    print(f"Step 2 result saved to: {OUTPUT_FILE.resolve()}")
    print(processing_summary.to_string(index=False))
    print(target_summary.to_string(index=False))

if __name__ == "__main__": main()