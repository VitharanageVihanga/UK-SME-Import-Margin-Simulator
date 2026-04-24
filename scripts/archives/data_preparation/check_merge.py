import pandas as pd
df = pd.read_csv("/Users/vihangasathsara/Desktop/fyp-project/data/output/merged_hmrc_ons_totals.csv")
print("Missing ONS values:", df['import_value_million_gbp'].isna().sum())

