import pandas as pd

INPUT_FILE = "data/output/merged_hmrc_ons_totals.csv"
OUTPUT_FILE = "data/output/ons_coverage_by_country_aggregated.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

agg = (
    df.groupby(["country_code", "year"])
    .agg(
        hmrc_rows=("value", "count"),
        ons_present=("import_value_million_gbp", lambda x: x.notna().any())
    )
    .reset_index()
)

coverage = (
    agg.groupby("country_code")
    .agg(
        total_years=("year", "count"),
        ons_covered_years=("ons_present", "sum")
    )
    .reset_index()
)

coverage["ons_coverage_pct"] = (
    coverage["ons_covered_years"]
    / coverage["total_years"]
    * 100
)

coverage = coverage.sort_values("ons_coverage_pct")

coverage.to_csv(OUTPUT_FILE, index=False)

print("Saved aggregated country ONS coverage →", OUTPUT_FILE)
print(coverage.head(10))
