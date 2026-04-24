import pandas as pd
import os

# ================== PATHS ==================
INPUT_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/output/merged_hmrc_ons_commodity.csv"
OUTPUT_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/output/commodity_summary.csv"

# ================== LOAD ==================
df = pd.read_csv(INPUT_FILE, low_memory=False)
print("Loaded rows:", len(df))
print("Columns BEFORE fix:", list(df.columns))

# ================== FIX COMMODITY COLUMN ==================
# HMRC is the base truth → use commodity_x
if "commodity_x" not in df.columns:
    raise ValueError("commodity_x missing — merge logic is broken upstream")

df["commodity"] = df["commodity_x"]

# Drop rows with no commodity
df = df[df["commodity"].notna()]

print("Columns AFTER fix:", list(df.columns))

# ================== AGGREGATE ==================
summary = (
    df.groupby(
        ["country_code", "commodity", "year"],
        as_index=False
    )
    .agg(
        hmrc_import_value=("value", "sum"),
        hmrc_net_mass=("net_mass", "sum"),
        hmrc_transactions=("value", "count"),
        ons_import_value=("import_value_million_gbp", "mean"),
    )
)

# ================== ONS UNIT FIX ==================
summary["ons_import_value_gbp"] = summary["ons_import_value"] * 1_000_000

# ================== GAP CALCULATIONS ==================
summary["value_gap"] = (
    summary["hmrc_import_value"] - summary["ons_import_value_gbp"]
)

summary["value_gap_pct"] = (
    summary["value_gap"] / summary["ons_import_value_gbp"]
) * 100

# ================== DATA QUALITY FLAGS ==================
summary["data_quality_flag"] = "OK"

summary.loc[
    summary["ons_import_value_gbp"].isna(),
    "data_quality_flag"
] = "NO_ONS_DATA"

summary.loc[
    summary["hmrc_transactions"] < 10,
    "data_quality_flag"
] = "LOW_SAMPLE_SIZE"

summary.loc[
    summary["value_gap_pct"].abs() > 100,
    "data_quality_flag"
] = "EXTREME_GAP"

# Cap insane percentages for charts only
summary["value_gap_pct"] = summary["value_gap_pct"].clip(-500, 500)

# ================== SAVE ==================
summary.to_csv(OUTPUT_FILE, index=False)

print("Saved commodity summary →", OUTPUT_FILE)
print(summary.head())
print("Final row count:", len(summary))
