"""Step 1: high-precision audit of person linkage across adjacent years."""
from pathlib import Path
import re
import unicodedata
import pandas as pd

INPUT_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("person_step_1_linkage_audit.xlsx")
MARKERS = ["vp","svp","evp","sevp","dir","sdir","md","smd","se","vc","svc","president","board","ceo","cxo","primary","support","bu"]
REQUIRED = ["company","GV_KEY","year","full_name","first_name","last_name","uniqueid",*MARKERS]

def norm(value):
    if pd.isna(value): return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", value)

def main():
    if not INPUT_FILE.exists(): raise FileNotFoundError(f"Put {INPUT_FILE.name} next to this script.")
    df = pd.read_excel(INPUT_FILE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    invalid = []
    for c in MARKERS:
        bad = df[c].notna() & ~df[c].isin([0, 1])
        invalid.append(pd.DataFrame({"row": df.index[bad], "field": c, "value": df.loc[bad, c]}))
    invalid = pd.concat(invalid, ignore_index=True)
    df["org_key"] = df["GV_KEY"].astype("string").str.strip()
    df["name_key"] = df["full_name"].map(norm)
    df["person_key"] = df["org_key"].fillna("") + "|" + df["name_key"]
    usable = df[df["org_key"].notna() & df["name_key"].ne("")].copy()
    py = usable.groupby(["person_key", "year"], dropna=False).agg(
        company=("company", "first"), GV_KEY=("GV_KEY", "first"), full_name=("full_name", "first"),
        n_source_rows=("uniqueid", "size"), n_companies=("company", "nunique")
    ).reset_index()
    py["linkage_eligible"] = py["n_source_rows"].eq(1) & py["n_companies"].eq(1)
    eligible = py[py["linkage_eligible"]].copy()
    current = eligible.copy(); current["next_year"] = current["year"] + 1
    next_state = eligible[["person_key", "year", "company", "GV_KEY", "full_name"]].rename(
        columns={"year":"next_year", "company":"next_company", "GV_KEY":"next_GV_KEY", "full_name":"next_full_name"}
    )
    pairs = current.merge(next_state, on=["person_key", "next_year"], how="inner")
    company_gv = df.groupby("company")["GV_KEY"].nunique(dropna=True).reset_index(name="n_gv_keys")
    company_gv = company_gv[company_gv["n_gv_keys"] > 1]
    summary = pd.DataFrame({"metric":["input_rows","usable_rows","person_year_records","eligible_person_year","adjacent_eligible_pairs","ambiguous_person_year","invalid_marker_values"],
        "value":[len(df),len(usable),len(py),len(eligible),len(pairs),int((~py["linkage_eligible"]).sum()),len(invalid)]})
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        pairs.to_excel(writer, sheet_name="adjacent_pairs", index=False)
        eligible.to_excel(writer, sheet_name="eligible_person_year", index=False)
        py[~py["linkage_eligible"]].to_excel(writer, sheet_name="excluded_person_year", index=False)
        invalid.to_excel(writer, sheet_name="invalid_marker_values", index=False)
        company_gv.to_excel(writer, sheet_name="company_gv_conflicts", index=False)
    print(f"Step 1 result saved to: {OUTPUT_FILE.resolve()}")
    print(summary.to_string(index=False))

if __name__ == "__main__": main()
