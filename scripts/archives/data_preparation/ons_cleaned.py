import pandas as pd

# ------------ PATHS ------------
ONS_TOTALS = "/Users/vihangasathsara/Desktop/fyp-project/data/raw/ons/ons_country_totals.xlsx"
ONS_COMMODITY = "/Users/vihangasathsara/Desktop/fyp-project/data/raw/ons/ons_country_by_commodity.xlsx"



# ------------ CLEAN COUNTRY TOTALS ------------
def clean_ons_country_totals():
    df = pd.read_excel(ONS_TOTALS, sheet_name="2. Annual Imports", skiprows=3)

    df = df.rename(columns={
        "Country Code": "country_code",
        "Country Name": "country_name"
    })

    df = df[~df["country_name"].astype(str).str.contains("Total", case=False, na=False)]

    df_long = df.melt(
        id_vars=["country_code", "country_name"],
        var_name="year",
        value_name="import_value_million_gbp"
    )

    df_long = df_long.dropna(subset=["import_value_million_gbp"])
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long[df_long["import_value_million_gbp"] > 0]


    out = "ons_country_totals_clean.csv"
    df_long.to_csv(out, index=False)
    print(f"✔ Saved cleaned country totals → {out}  ({len(df_long)} rows)")



# ------------ CLEAN COUNTRY BY COMMODITY ------------
def clean_ons_by_commodity():
    df = pd.read_excel(ONS_COMMODITY, sheet_name="1. Annual Imports", skiprows=3)

    df = df.rename(columns={
        "COMMODITY": "commodity",
        "COUNTRY": "country_name",
        "DIRECTION": "direction"
    })

    df = df[df["direction"] == "IM Imports"]
    df = df[~df["country_name"].astype(str).str.contains("Total", case=False, na=False)]

    df_long = df.melt(
        id_vars=["commodity", "country_name", "direction"],
        var_name="year",
        value_name="import_value_million_gbp"
    )

    df_long = df_long.dropna(subset=["import_value_million_gbp"])
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long[df_long["import_value_million_gbp"] > 0]


    out = "ons_country_by_commodity_clean.csv"
    df_long.to_csv(out, index=False)
    print(f"✔ Saved cleaned commodity imports → {out}  ({len(df_long)} rows)")



# ------------ MAIN ------------
def main():
    print("\n==== CLEANING ONS DATA ====\n")
    clean_ons_country_totals()
    clean_ons_by_commodity()
    print("\n🎉 ALL ONS CLEANING COMPLETED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
