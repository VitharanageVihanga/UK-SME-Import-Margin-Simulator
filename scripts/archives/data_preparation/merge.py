import pandas as pd
import os

# ======================================
# FILE PATHS  (EDIT ONLY THESE)
# ======================================
HMRC_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/processed/hmrc_cleaned.csv"
ONS_TOTALS_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/processed/ons_country_totals_clean.csv"
ONS_COMMODITY_FILE = "/Users/vihangasathsara/Desktop/fyp-project/data/processed/ons_country_by_commodity_clean.csv"

OUTPUT_FOLDER = "/Users/vihangasathsara/Desktop/fyp-project/data/output/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ======================================
# NORMALISE COLUMNS
# ======================================
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


# ======================================
# LOAD ALL CLEANED DATA
# ======================================
def load_data():
    hmrc = pd.read_csv(HMRC_FILE, low_memory=False)
    ons_totals = pd.read_csv(ONS_TOTALS_FILE)
    ons_commodity = pd.read_csv(ONS_COMMODITY_FILE)

    print("Loaded shapes:")
    print("HMRC:", hmrc.shape)
    print("ONS totals:", ons_totals.shape)
    print("ONS commodity:", ons_commodity.shape)

    return hmrc, ons_totals, ons_commodity


# ======================================
# PREPARE DATA (NORMALISE + FIX CODE ISSUES)
# ======================================
def prepare_data():
    hmrc, ons_totals, ons_commodity = load_data()

    # Standardise column names
    hmrc = normalize_columns(hmrc)
    ons_totals = normalize_columns(ons_totals)
    ons_commodity = normalize_columns(ons_commodity)

    # HMRC fix: rename partner_country → country_code
    if "partner_country" in hmrc.columns:
        hmrc = hmrc.rename(columns={"partner_country": "country_code"})

    # ONS fixes
    ons_totals = ons_totals.rename(columns={
        "countrycode": "country_code",
        "country": "country_name"
    })

    ons_commodity = ons_commodity.rename(columns={
        "country": "country_name"
    })

    # ======================================
    # FIX 1 — Remove garbage country codes like YY, ZZ, XX
    # ======================================
    bad_codes = ["YY", "ZZ", "XX", None, "", "UNK"]
    hmrc = hmrc[~hmrc["country_code"].isin(bad_codes)]

    # ======================================
    # FIX 2 — Replace inconsistent codes (e.g., EL → GR)
    # ======================================
    code_map = {
        "EL": "GR"  # Greece naming mismatch
    }
    hmrc["country_code"] = hmrc["country_code"].replace(code_map)

    # ======================================
    # FIX 3 — Filter HMRC to only countries that appear in ONS totals
    # ======================================
    valid_codes = ons_totals["country_code"].unique()
    hmrc = hmrc[hmrc["country_code"].isin(valid_codes)]

    print("\nAfter cleaning:")
    print("HMRC:", hmrc.shape)
    print("ONS totals:", ons_totals.shape)
    print("ONS commodity:", ons_commodity.shape)

    return hmrc, ons_totals, ons_commodity


# ======================================
# MERGE EVERYTHING
# ======================================
def merge_all(hmrc, ons_totals, ons_commodity):

    print("\n=== STEP 1: Merge HMRC with ONS totals ===")
    merged_totals = hmrc.merge(
        ons_totals,
        on=["country_code", "year"],
        how="left"
    )
    merged_totals.to_csv(OUTPUT_FOLDER + "merged_hmrc_ons_totals.csv", index=False)
    print("Saved: merged_hmrc_ons_totals.csv")

    print("\n=== STEP 2: Add country_code to ONS commodity (required!) ===")
    lookup = ons_totals[["country_code", "country_name"]].drop_duplicates()
    ons_commodity = ons_commodity.merge(lookup, on="country_name", how="left")

    print("\n=== STEP 3: Merge HMRC with ONS commodity ===")
    merged_com = hmrc.merge(
        ons_commodity,
        on=["country_code", "year"],
        how="left"
    )
    merged_com.to_csv(OUTPUT_FOLDER + "merged_hmrc_ons_commodity.csv", index=False)
    print("Saved: merged_hmrc_ons_commodity.csv")

    print("\n==== ALL MERGES COMPLETED SUCCESSFULLY ====\n")


# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    hmrc, ons_totals, ons_commodity = prepare_data()
    merge_all(hmrc, ons_totals, ons_commodity)


