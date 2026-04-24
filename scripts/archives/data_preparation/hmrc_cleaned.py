import os
import pandas as pd

HMRC_FOLDER = "/Users/vihangasathsara/Desktop/fyp-project/data/raw/hmrc"


# Correct fixed-width layout based on your 3 sample rows
colspecs = [
    (0, 4),    # year
    (4, 6),    # month
    (6, 8),    # day
    (8, 15),   # trader_id
    (15, 21),  # placeholder1
    (21, 23),  # flow
    (23, 26),  # mode
    (26, 29),  # commodity
    (29, 31),  # partner_country
    (31, 37),  # placeholder2
    (37, 44),  # padding
    (44, 57),  # value
    (57, 70),  # net_mass
    (70, 83),  # sup_units
    (83, 87),  # flag
]

colnames = [
    "year", "month", "day", "trader_id",
    "placeholder1", "flow", "mode",
    "commodity", "partner_country",
    "placeholder2", "padding",
    "value", "net_mass", "sup_units",
    "flag"
]

def read_hmrc_file(path):
    df = pd.read_fwf(path, colspecs=colspecs, names=colnames)

    # Remove useless fields
    df = df.drop(columns=["placeholder1", "placeholder2", "padding"])

    # Convert numeric fields
    for col in ["value", "net_mass", "sup_units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def main():
    files = sorted([f for f in os.listdir(HMRC_FOLDER) if f.endswith(".txt")])
    print(f"Found {len(files)} HMRC files.\n")

    all_dfs = []

    for f in files:
        fp = os.path.join(HMRC_FOLDER, f)
        df = read_hmrc_file(fp)
        print(f"{f}: {len(df)} rows")

        all_dfs.append(df)

    final = pd.concat(all_dfs, ignore_index=True)

    final.to_csv("hmrc_cleaned.csv", index=False)
    print("\nSaved → hmrc_cleaned.csv")

if __name__ == "__main__":
    main()
