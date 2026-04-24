"""
Data Merge - merges HMRC and ONS data with correct commodity harmonisation.

A few things that needed fixing from the original:
- ONS country format is like "AE United Arab Emirates" - need to pull the code out
- ONS uses SITC sections (0-9) while HMRC uses HS chapters (01-99)
- SITC sub-categories like "00 Live animals" need to map back to section 0
"""
import logging
import os
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# file paths
HMRC_FILE = "data/processed/hmrc_cleaned.csv"
ONS_TOTALS_FILE = "data/processed/ons_country_totals_clean.csv"
ONS_COMMODITY_FILE = "data/processed/ons_country_by_commodity_clean.csv"
OUTPUT_FOLDER = "data/output/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ONS uses SITC sections (0-9), HMRC uses HS chapters (01-99) - this maps between them
HS2_TO_SITC = {
    # food and live animals
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
    21: 0, 22: 1, 23: 0, 24: 1,
    # crude materials / fuels
    25: 2, 26: 2, 27: 3,
    # chemicals
    28: 5, 29: 5, 30: 5, 31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5,
    # manufactured goods
    39: 6, 40: 6, 41: 2, 42: 8, 43: 8,
    44: 6, 45: 6, 46: 6, 47: 6, 48: 6, 49: 8, 50: 6,
    51: 6, 52: 6, 53: 6, 54: 6, 55: 6, 56: 6, 57: 6, 58: 6, 59: 6,
    60: 6, 61: 8, 62: 8, 63: 8, 64: 8, 65: 8, 66: 6, 67: 6,
    68: 6, 69: 6, 70: 6, 71: 8, 72: 6, 73: 6, 74: 6, 75: 6, 76: 6,
    78: 6, 79: 6, 80: 6, 81: 6, 82: 6, 83: 6,
    # machinery and transport
    84: 7, 85: 7, 86: 7, 87: 7, 88: 7, 89: 7,
    # miscellaneous manufactures
    90: 8, 91: 8, 92: 8, 93: 8, 94: 8, 95: 8, 96: 8, 97: 8,
    # other
    99: 9,
}

SITC_NAMES = {
    0: "0 Food & live animals",
    1: "1 Beverages & tobacco",
    2: "2 Crude materials",
    3: "3 Fuels",
    4: "4 Animal & vegetable oils",
    5: "5 Chemicals",
    6: "6 Manufactured goods",
    7: "7 Machinery & transport equipment",
    8: "8 Miscellaneous manufactures",
    9: "9 Other commodities",
}


def extract_country_code_from_name(country_name):
    """Pulls the 2-letter country code out of ONS names like 'AE United Arab Emirates'."""
    if pd.isna(country_name):
        return None
    parts = str(country_name).split(' ', 1)
    if len(parts) >= 1 and len(parts[0]) == 2 and parts[0].isupper():
        return parts[0]
    return None


def extract_main_sitc_section(sitc_name):
    """
    Gets the top-level SITC section (0-9) from ONS commodity strings.
    e.g. '00 Live animals' -> 0, '792 Aircraft' -> 7, 'T Total' -> None
    """
    if pd.isna(sitc_name):
        return None
    
    name_str = str(sitc_name).strip()
    
    # skip the aggregate 'T Total' rows
    if name_str.startswith('T ') or name_str == 'T Total':
        return None

    # first digit of the number is the SITC section
    match = re.match(r'^(\d+)', name_str)
    if match:
        num_str = match.group(1)
        # The main section is the first digit
        return int(num_str[0])
    
    return None


def normalize_columns(df):
    """Lowercases and underscores column names for consistency."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_data():
    """Load all cleaned data files."""
    logger.info("=" * 60)
    logger.info("LOADING DATA FILES")
    logger.info("=" * 60)
    
    hmrc = pd.read_csv(HMRC_FILE, low_memory=False)
    ons_totals = pd.read_csv(ONS_TOTALS_FILE)
    ons_commodity = pd.read_csv(ONS_COMMODITY_FILE)
    
    hmrc = normalize_columns(hmrc)
    ons_totals = normalize_columns(ons_totals)
    ons_commodity = normalize_columns(ons_commodity)
    
    logger.info(f"HMRC rows: {len(hmrc):,}")
    logger.info(f"ONS totals rows: {len(ons_totals):,}")
    logger.info(f"ONS commodity rows: {len(ons_commodity):,}")
    
    return hmrc, ons_totals, ons_commodity


def prepare_hmrc(hmrc):
    """Prepare HMRC data with HS2 chapter extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING HMRC DATA")
    logger.info("=" * 60)
    
    # Rename partner_country to country_code
    if "partner_country" in hmrc.columns:
        hmrc = hmrc.rename(columns={"partner_country": "country_code"})
    
    # Remove bad country codes
    bad_codes = ["YY", "ZZ", "XX", "", "UNK"]
    original_len = len(hmrc)
    hmrc = hmrc[~hmrc["country_code"].isin(bad_codes)]
    hmrc = hmrc[hmrc["country_code"].notna()]
    logger.info(f"Removed {original_len - len(hmrc):,} rows with invalid country codes")
    
    # Extract HS2 chapter (first 2 digits of commodity code)
    def get_hs2_chapter(commodity_code):
        try:
            code = int(commodity_code)
            if code < 100:
                return code
            return int(str(code)[:2])
        except (ValueError, TypeError):
            return None
    
    hmrc["hs2_chapter"] = hmrc["commodity"].apply(get_hs2_chapter)
    
    # Map to SITC section
    hmrc["sitc_section"] = hmrc["hs2_chapter"].apply(lambda x: HS2_TO_SITC.get(x, None))
    hmrc["sitc_name"] = hmrc["sitc_section"].apply(lambda x: SITC_NAMES.get(x, None))
    
    logger.info(f"HMRC unique HS2 chapters: {hmrc['hs2_chapter'].nunique()}")
    logger.info(f"HMRC unique SITC sections: {hmrc['sitc_section'].nunique()}")
    logger.info(f"HMRC year range: {hmrc['year'].min()} - {hmrc['year'].max()}")
    
    return hmrc


def prepare_ons_commodity(ons_commodity):
    """Prepare ONS commodity data with extracted country codes and SITC sections."""
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING ONS COMMODITY DATA")
    logger.info("=" * 60)
    
    # Extract country code from country_name (e.g., "AE United Arab Emirates" -> "AE")
    ons_commodity["country_code"] = ons_commodity["country_name"].apply(extract_country_code_from_name)
    
    # Remove rows without valid country code
    original_len = len(ons_commodity)
    ons_commodity = ons_commodity[ons_commodity["country_code"].notna()]
    logger.info(f"Removed {original_len - len(ons_commodity):,} rows without valid country code")
    
    # Rename commodity column for clarity
    ons_commodity = ons_commodity.rename(columns={"commodity": "sitc_name_raw"})
    
    # Extract main SITC section (0-9) from the commodity name
    ons_commodity["sitc_section"] = ons_commodity["sitc_name_raw"].apply(extract_main_sitc_section)
    ons_commodity["sitc_name"] = ons_commodity["sitc_section"].apply(lambda x: SITC_NAMES.get(x, None))
    
    # Remove rows without valid SITC section (e.g., "T Total")
    before_filter = len(ons_commodity)
    ons_commodity = ons_commodity[ons_commodity["sitc_section"].notna()]
    logger.info(f"Removed {before_filter - len(ons_commodity):,} rows without valid SITC section")
    
    logger.info(f"ONS unique SITC sections: {ons_commodity['sitc_section'].nunique()}")
    logger.info(f"ONS unique countries: {ons_commodity['country_code'].nunique()}")
    logger.info(f"ONS year range: {ons_commodity['year'].min()} - {ons_commodity['year'].max()}")
    
    # Show unique main SITC sections found
    logger.info("\nONS Main SITC sections found:")
    for section in sorted(ons_commodity["sitc_section"].unique()):
        name = SITC_NAMES.get(section, "Unknown")
        logger.info(f"  {section} -> {name}")
    
    return ons_commodity


def compute_ons_coverage_by_sitc(ons_commodity):
    """
    Compute ONS coverage at SITC section level.
    Coverage = which SITC sections have data for which years.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING ONS COVERAGE BY SITC SECTION")
    logger.info("=" * 60)
    
    # Aggregate to main SITC section level (summing all sub-categories)
    sitc_year_agg = (
        ons_commodity.groupby(["sitc_section", "sitc_name", "year"])
        .agg(
            total_value=("import_value_million_gbp", "sum"),
            country_count=("country_code", "nunique"),
            raw_category_count=("sitc_name_raw", "nunique")
        )
        .reset_index()
    )
    
    # Mark as covered if there's meaningful data
    sitc_year_agg["has_data"] = sitc_year_agg["total_value"] > 0
    
    # Compute coverage per SITC section
    all_years = sorted(ons_commodity["year"].unique())
    total_years = len(all_years)
    
    sitc_coverage = (
        sitc_year_agg.groupby(["sitc_section", "sitc_name"])
        .agg(
            years_with_data=("has_data", "sum"),
            total_value=("total_value", "sum")
        )
        .reset_index()
    )
    
    sitc_coverage["total_years"] = total_years
    sitc_coverage["coverage_pct"] = (sitc_coverage["years_with_data"] / total_years * 100).round(1)
    
    logger.info(f"\nTotal years in ONS data: {total_years} ({min(all_years)}-{max(all_years)})")
    logger.info("\nSITC Section Coverage Summary:")
    for _, row in sitc_coverage.sort_values("sitc_section").iterrows():
        logger.info(f"  Section {int(row['sitc_section'])}: {row['sitc_name']} - {row['coverage_pct']:.1f}% ({int(row['years_with_data'])}/{total_years} years)")
    
    return sitc_coverage, total_years


def create_hs2_coverage_from_sitc(sitc_coverage, total_years):
    """
    Map SITC coverage to HS2 chapter coverage.
    Each HS2 chapter inherits coverage from its parent SITC section.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MAPPING SITC COVERAGE TO HS2 CHAPTERS")
    logger.info("=" * 60)
    
    hs2_coverage_rows = []
    
    for hs2, sitc_section in HS2_TO_SITC.items():
        sitc_row = sitc_coverage[sitc_coverage["sitc_section"] == sitc_section]
        
        if len(sitc_row) > 0:
            row = sitc_row.iloc[0]
            coverage_pct = row["coverage_pct"]
            sitc_name = row["sitc_name"]
            years_covered = int(row["years_with_data"])
        else:
            # No ONS data for this SITC section
            coverage_pct = 0.0
            sitc_name = SITC_NAMES.get(sitc_section, "Unknown")
            years_covered = 0
        
        # Classify coverage
        if coverage_pct >= 80:
            coverage_class = "High coverage"
        elif coverage_pct >= 50:
            coverage_class = "Partial coverage"
        elif coverage_pct > 0:
            coverage_class = "Low coverage"
        else:
            coverage_class = "No coverage"
        
        hs2_coverage_rows.append({
            "commodity": hs2,
            "hs2_chapter": hs2,
            "sitc_section": sitc_section,
            "sitc_category": sitc_name,
            "total_years": total_years,
            "ons_covered_years": years_covered,
            "ons_coverage_pct": coverage_pct,
            "coverage_class": coverage_class
        })
    
    hs2_coverage = pd.DataFrame(hs2_coverage_rows)
    hs2_coverage = hs2_coverage.sort_values("commodity")
    
    # Summary
    logger.info("\nHS2 Coverage Distribution:")
    logger.info(hs2_coverage["coverage_class"].value_counts())
    
    logger.info("\nSample HS2 to SITC mappings:")
    for _, row in hs2_coverage.head(10).iterrows():
        logger.info(f"  HS {int(row['commodity']):02d} -> Section {int(row['sitc_section'])} ({row['sitc_category']}) = {row['coverage_class']}")
    
    return hs2_coverage


def merge_hmrc_ons_totals(hmrc, ons_totals):
    """Merge HMRC with ONS country totals (lightweight)."""
    logger.info("\n" + "=" * 60)
    logger.info("MERGING HMRC WITH ONS TOTALS")
    logger.info("=" * 60)
    
    # Aggregate HMRC to country-year level for summary
    hmrc_agg = (
        hmrc.groupby(["country_code", "year"])
        .agg(hmrc_total_value=("value", "sum"))
        .reset_index()
    )
    
    merged = hmrc_agg.merge(
        ons_totals,
        on=["country_code", "year"],
        how="left"
    )
    
    match_rate = merged["import_value_million_gbp"].notna().mean() * 100
    logger.info(f"Match rate: {match_rate:.1f}%")
    logger.info(f"Rows: {len(merged):,}")
    
    output_file = OUTPUT_FOLDER + "merged_hmrc_ons_totals.csv"
    merged.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file}")
    
    return merged


def save_coverage_output(hs2_coverage):
    """Save the HS2 coverage classification."""
    logger.info("\n" + "=" * 60)
    logger.info("SAVING COVERAGE OUTPUT")
    logger.info("=" * 60)
    
    output_file = OUTPUT_FOLDER + "ons_coverage_by_commodity_classified.csv"
    hs2_coverage.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file}")
    
    # Also save aggregated version
    agg_file = OUTPUT_FOLDER + "ons_coverage_by_commodity_aggregated.csv"
    hs2_coverage.to_csv(agg_file, index=False)
    logger.info(f"Saved: {agg_file}")


def main():
    """Main pipeline execution."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA MERGE PIPELINE - FIXED VERSION")
    logger.info("=" * 60)
    
    # Load data
    hmrc, ons_totals, ons_commodity = load_data()
    
    # Prepare datasets
    hmrc = prepare_hmrc(hmrc)
    ons_commodity = prepare_ons_commodity(ons_commodity)
    
    # Compute SITC-level coverage from ONS
    sitc_coverage, total_years = compute_ons_coverage_by_sitc(ons_commodity)
    
    # Map to HS2 chapters
    hs2_coverage = create_hs2_coverage_from_sitc(sitc_coverage, total_years)
    
    # Save coverage output
    save_coverage_output(hs2_coverage)
    
    # Merge HMRC with ONS totals (for summary stats)
    merge_hmrc_ons_totals(hmrc, ons_totals)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("\nOutput files:")
    logger.info(f"  - {OUTPUT_FOLDER}ons_coverage_by_commodity_classified.csv")
    logger.info(f"  - {OUTPUT_FOLDER}merged_hmrc_ons_totals.csv")


if __name__ == "__main__":
    main()
