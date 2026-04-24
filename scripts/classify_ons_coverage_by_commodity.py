import pandas as pd

# ================= FILE PATHS =================
INPUT_FILE = "data/output/ons_coverage_by_commodity_aggregated.csv"
OUTPUT_FILE = "data/output/ons_coverage_by_commodity_classified.csv"

# ================= LOAD DATA =================
df = pd.read_csv(INPUT_FILE)

# Sanity check
required_cols = {"commodity", "ons_coverage_pct"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ================= CLASSIFY COVERAGE =================
def classify_coverage(pct):
    if pct == 0:
        return "No coverage"
    elif pct <= 40:
        return "Low coverage"
    elif pct <= 80:
        return "Partial coverage"
    else:
        return "High coverage"

df["coverage_class"] = df["ons_coverage_pct"].apply(classify_coverage)

# ================= SAVE =================
df.to_csv(OUTPUT_FILE, index=False)

print("Saved classified ONS coverage by commodity →", OUTPUT_FILE)
print(df[["commodity", "ons_coverage_pct", "coverage_class"]].head(10))
print("Total commodities:", len(df))
