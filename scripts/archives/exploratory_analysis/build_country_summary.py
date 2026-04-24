import pandas as pd
import os

# ========= FILE PATHS =========
MERGED_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/output/merged_hmrc_ons_totals.csv"
OUTPUT_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/output/country_summary.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ========= LOAD DATA =========
df = pd.read_csv(MERGED_FILE, low_memory=False)

print("Loaded rows:", len(df))

# ========= BASIC CLEANING =========
# Drop invalid / unknown country codes
df = df[df["country_code"].notna()]
df = df[df["country_code"] != "YY"]

# ========= AGGREGATE TO COUNTRY–YEAR LEVEL =========
summary = (
    df.groupby(["country_code", "country_name", "year"])
      .agg(
          hmrc_import_value=("value", "sum"),
          hmrc_net_mass=("net_mass", "sum"),
          hmrc_transactions=("value", "count"),
          ons_import_value=("import_value_million_gbp", "sum")
      )
      .reset_index()
)

# ========= UNIT CONVERSION =========
# ONS values are in million GBP → convert to GBP
summary["ons_import_value_gbp"] = summary["ons_import_value"] * 1_000_000

# ========= SAFETY FILTERS =========
# Remove tiny ONS denominators (avoids insane % values)
summary = summary[summary["ons_import_value_gbp"] > 1_000_000]

# ========= GAP CALCULATIONS =========
summary["value_gap"] = (
    summary["hmrc_import_value"] - summary["ons_import_value_gbp"]
)

summary["value_gap_pct"] = (
    summary["value_gap"] / summary["ons_import_value_gbp"]
) * 100

# Cap extreme percentages for reporting sanity
summary["value_gap_pct"] = summary["value_gap_pct"].clip(-500, 500)

# ========= DATA QUALITY FLAGS =========
summary["data_quality_flag"] = "OK"

summary.loc[
    summary["ons_import_value_gbp"] < 5_000_000,
    "data_quality_flag"
] = "LOW_ONS_BASE"

summary.loc[
    summary["value_gap_pct"].abs() > 100,
    "data_quality_flag"
] = "EXTREME_GAP"

# ========= SAVE OUTPUT =========
summary.to_csv(OUTPUT_FILE, index=False)

print("Saved country summary →", OUTPUT_FILE)
print(summary.head())

