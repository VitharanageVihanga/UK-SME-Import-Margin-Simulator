import pandas as pd

# ================== FILE PATHS ==================
INPUT_FILE = "data/output/merged_hmrc_ons_commodity.csv"
OUTPUT_FILE = "data/output/ons_coverage_by_commodity_aggregated.csv"

# ================== LOAD ==================
df = pd.read_csv(INPUT_FILE, low_memory=False)

# ================== FIX COMMODITY COLUMN ==================
# HMRC commodity is commodity_x — this is intentional
if "commodity_x" not in df.columns:
    raise ValueError("commodity_x column missing — merge is broken")

df = df.rename(columns={"commodity_x": "commodity"})

# ================== ONS PRESENCE FLAG ==================
# ONS is considered present if import_value_million_gbp exists
df["ons_present"] = df["import_value_million_gbp"].notna()

# ================== YEARLY PRESENCE PER COMMODITY ==================
yearly = (
    df.groupby(["commodity", "year"])
      .agg(ons_present=("ons_present", "max"))
      .reset_index()
)

# ================== AGGREGATE TO COMMODITY LEVEL ==================
coverage = (
    yearly.groupby("commodity")
    .agg(
        total_years=("year", "nunique"),
        ons_covered_years=("ons_present", "sum")
    )
    .reset_index()
)

# ================== COVERAGE % ==================
coverage["ons_coverage_pct"] = (
    coverage["ons_covered_years"] / coverage["total_years"] * 100
)

# ================== SORT: WORST FIRST ==================
coverage = coverage.sort_values("ons_coverage_pct")

# ================== SAVE ==================
coverage.to_csv(OUTPUT_FILE, index=False)

print("Saved aggregated ONS coverage by commodity →", OUTPUT_FILE)
print(coverage.head(10))
print("Total commodities:", len(coverage))
