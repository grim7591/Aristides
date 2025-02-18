# %%
import pandas as pd
all_exclusions = pd.read_csv('Data/All_Excluded_Pids.csv')
XXI_exclusions = pd.read_csv('Data/21_excluded_sales.csv')
XXII_exclusions = pd.read_csv('Data/22_excluded_sales.csv')
XXIII_exclusions = pd.read_csv('Data/23_excluded_sales.csv')
XIV_exclusions = pd.read_csv('Data/24_excluded_sales.csv')
# %%
# Concatenate them all into one DataFrame
df_all = pd.concat([
    XXI_exclusions, 
    XXII_exclusions, 
    XXIII_exclusions, 
    XIV_exclusions
], ignore_index=True)

# Count how many times each prop_id appears
prop_counts = df_all['prop_id'].value_counts()

# Keep only the rows with a prop_id that appears more than once
df_more_than_one = df_all[df_all['prop_id'].isin(prop_counts.index[prop_counts > 1])]

# df_more_than_one now has all rows for prop_ids found in multiple tables
print(df_more_than_one)
# Write out the DataFrame to a CSV file
df_more_than_one.to_csv('Data/props_appearing_in_multiple_tables.csv', index=False)

print("Saved df_more_than_one to CSV.")# %%

# %%
# 1. Sort by prop_id and prop_val_yr so the "last" row is the one with the highest prop_val_yr
df_sorted = df_more_than_one.sort_values(["prop_id", "prop_val_yr"])

# 2. Drop duplicates on prop_id (keeping only the last row = highest prop_val_yr in each group)
df_most_recent = df_sorted.drop_duplicates(subset=["prop_id"], keep="last")

# 3. From these most-recent rows, find which prop_ids have sl_county_ratio_cd == 2
prop_ids_with_ratio_2 = df_most_recent.loc[
    df_most_recent["sl_county_ratio_cd"] == 2, 
    "prop_id"
].unique()

# 4. Filter the original df_more_than_one to keep ALL rows with those prop_ids
df_filtered = df_more_than_one[
    df_more_than_one["prop_id"].isin(prop_ids_with_ratio_2)
]

# Done! df_filtered now contains all rows for prop_ids
# where the most recent (highest prop_val_yr) row had sl_county_ratio_cd = 2.

# If desired, write it out to CSV
df_filtered.to_csv("Data/props_with_ratio2_most_recent.csv", index=False)

print("Created df_filtered with all rows for prop_ids whose most recent sl_county_ratio_cd is 2.")
# %%
# 1. Determine the prop_ids you need to remove (those in df_filtered)
prop_ids_to_remove = df_filtered['prop_id'].unique()

# 2. Remove these prop_ids from each exclusion list, creating new DataFrames with an "f" suffix
XXI_exclusions_f = XXI_exclusions[~XXI_exclusions['prop_id'].isin(prop_ids_to_remove)]
XXII_exclusions_f = XXII_exclusions[~XXII_exclusions['prop_id'].isin(prop_ids_to_remove)]
XXIII_exclusions_f = XXIII_exclusions[~XXIII_exclusions['prop_id'].isin(prop_ids_to_remove)]
XIV_exclusions_f = XIV_exclusions[~XIV_exclusions['prop_id'].isin(prop_ids_to_remove)]

# 3. Concatenate the four new DataFrames, then find prop_ids that appear more than once
df_allf = pd.concat([
    XXI_exclusions_f, 
    XXII_exclusions_f, 
    XXIII_exclusions_f, 
    XIV_exclusions_f
], ignore_index=True)

prop_counts_f = df_allf['prop_id'].value_counts()
df_more_than_onef = df_allf[df_allf['prop_id'].isin(prop_counts_f.index[prop_counts_f > 1])]

# 4. Write out df_more_than_onef to a *different* CSV filename
df_more_than_onef.to_csv('Data/props_appearing_in_multiple_tables_after_filter.csv', index=False)

print("Created df_more_than_onef and saved it to CSV.")

# %%
XXI_exclusions_f.to_csv('Data/XXI_exclusions_f.csv')
XXII_exclusions_f.to_csv('Data/XXII_exclusions_f.csv')
XXIII_exclusions_f.to_csv('Data/XXIII_exclusions_f.csv')
XIV_exclusions_f.to_csv('Data/XXIV_exclusions_f.csv')
# %%
# 1. Load the CSVs
XXI_exclusions_f = pd.read_csv('Data/XXI_exclusions_f.csv')
XXII_exclusions_f = pd.read_csv('Data/XXII_exclusions_f.csv')
XXIII_exclusions_f = pd.read_csv('Data/XXIII_exclusions_f.csv')
XIV_exclusions_f = pd.read_csv('Data/XXIV_exclusions_f.csv')

# 2. Load the df_more_than_onef CSV
df_more_than_onef = pd.read_csv('Data/props_appearing_in_multiple_tables_after_filter.csv')

# 1. Extract the prop_ids in df_more_than_onef
pids_to_remove_f2 = df_more_than_onef['prop_id'].unique()

# 2. Remove them from each exclusions_f table
XXI_exclusions_f2 = XXI_exclusions_f[~XXI_exclusions_f['prop_id'].isin(pids_to_remove_f2)]
XXII_exclusions_f2 = XXII_exclusions_f[~XXII_exclusions_f['prop_id'].isin(pids_to_remove_f2)]
XXIII_exclusions_f2 = XXIII_exclusions_f[~XXIII_exclusions_f['prop_id'].isin(pids_to_remove_f2)]
XIV_exclusions_f2 = XIV_exclusions_f[~XIV_exclusions_f['prop_id'].isin(pids_to_remove_f2)]

# 3. Save them out to CSV
XXI_exclusions_f2.to_csv('Data/XXI_exclusions_f2.csv', index=False)
XXII_exclusions_f2.to_csv('Data/XXII_exclusions_f2.csv', index=False)
XXIII_exclusions_f2.to_csv('Data/XXIII_exclusions_f2.csv', index=False)
XIV_exclusions_f2.to_csv('Data/XXIV_exclusions_f2.csv', index=False)

print("Saved XXI_exclusions_f2, XXII_exclusions_f2, XXIII_exclusions_f2, XIV_exclusions_f2 to CSV.")

# %%
XXI_exclusions_f3=pd.read_csv('Data/XXI_exclusions_f2.csv')
XXII_exclusions_f3 = pd.read_csv('Data/XXII_exclusions_f2.csv')
XXIII_exclusions_f3 = pd.read_csv('Data/XXIII_exclusions_f2.csv')
XXIV_exclusions_f3 = pd.read_csv('Data/XXIV_exclusions_f2.csv')

hand_picked_outliers = pd.read_csv('Data/HandPickedOutliers.csv')

# Create a set of outlier_ids for fast membership checks
outlier_ids = set(hand_picked_outliers["prop_id"])

# Put your year/era exclusions dataframes in a dictionary
exclusion_data = {
    "XXI_exclusions_f3": XXI_exclusions_f3,
    "XXII_exclusions_f3": XXII_exclusions_f3,
    "XXIII_exclusions_f3": XXIII_exclusions_f3,
    "XXIV_exclusions_f3": XXIV_exclusions_f3
}

# For each DataFrame, count how many rows have a prop_id in outlier_ids
for name, df in exclusion_data.items():
    # .isin(...) checks if each prop_id is in the set
    # .sum() counts how many Trues
    count_outliers = df["prop_id"].isin(outlier_ids).sum()
    print(f"{name} contains {count_outliers} rows with outlier prop_id")
# %%
import pandas as pd

# 1) Read your "f2" CSV files
XXI_exclusions_f2  = pd.read_csv('Data/XXI_exclusions_f2.csv')
XXII_exclusions_f2 = pd.read_csv('Data/XXII_exclusions_f2.csv')
XXIII_exclusions_f2 = pd.read_csv('Data/XXIII_exclusions_f2.csv')
XXIV_exclusions_f2 = pd.read_csv('Data/XXIV_exclusions_f2.csv')

# 2) Load the hand-picked outliers and turn them into a set of IDs
hand_picked_outliers = pd.read_csv('Data/HandPickedOutliers.csv')
outlier_ids = set(hand_picked_outliers["prop_id"])  # ensures fast membership checks

# 3) Filter each exclusions DataFrame to keep only rows matching outlier_ids
XXI_exclusions_f3  = XXI_exclusions_f2[XXI_exclusions_f2["prop_id"].isin(outlier_ids)]
XXII_exclusions_f3 = XXII_exclusions_f2[XXII_exclusions_f2["prop_id"].isin(outlier_ids)]
XXIII_exclusions_f3 = XXIII_exclusions_f2[XXIII_exclusions_f2["prop_id"].isin(outlier_ids)]
XXIV_exclusions_f3 = XXIV_exclusions_f2[XXIV_exclusions_f2["prop_id"].isin(outlier_ids)]

# 4) Write the filtered DataFrames to new "f3" CSV files
XXI_exclusions_f3.to_csv('Data/XXI_exclusions_f3.csv', index=False)
XXII_exclusions_f3.to_csv('Data/XXII_exclusions_f3.csv', index=False)
XXIII_exclusions_f3.to_csv('Data/XXIII_exclusions_f3.csv', index=False)
XXIV_exclusions_f3.to_csv('Data/XXIV_exclusions_f3.csv', index=False)

print("New f3 CSV files have been created with only outlier prop_id rows.")

# %%
import pandas as pd

# --- 1) Load your existing DataFrames ---
MLS_SalesXXI   = pd.read_csv('Data/MLSData/2021MLSData.csv')
MLS_SalesXXII  = pd.read_csv('Data/MLSData/2022MLSData.csv')
MLS_SalesXXIII = pd.read_csv('Data/MLSData/2023MLSData.csv')
MLS_SalesXXIV  = pd.read_csv('Data/MLSData/2024MLSData.csv')

# --- 2) Combine all MLS data into a single DataFrame with an explicit "SaleYear" ---
#     Assuming each file is strictly for that year, we can add a column. 
#     (If your MLS files already have a year column, you can skip or adapt this part.)

MLS_SalesXXI["SaleYear"]   = 2021
MLS_SalesXXII["SaleYear"]  = 2022
MLS_SalesXXIII["SaleYear"] = 2023
MLS_SalesXXIV["SaleYear"]  = 2024

mls_combined = pd.concat([MLS_SalesXXI, MLS_SalesXXII, MLS_SalesXXIII, MLS_SalesXXIV],
                         ignore_index=True)

# --- 3) Compute the "sale year" we want from pids_to_remove_f2 (prop_val_yr - 1) ---
df_more_than_onef["SaleYearWanted"] = df_more_than_onef["prop_val_yr"] - 1

# --- 4) Merge pids_to_remove_f2 with the MLS data to see which properties match ---
#     We match "geo_id" in pids_to_remove_f2 with "Tax ID" in MLS (adjust if your MLS column differs)
#     and also match "SaleYearWanted" with "SaleYear" to confirm the property sold that year in MLS.

merged = df_more_than_onef.merge(
    mls_combined,
    how="inner",
    left_on=["geo_id", "SaleYearWanted"],
    right_on=["Tax ID", "SaleYear"]
)

# Now, 'merged' contains rows for any property-year combination that *does* exist in MLS.
# We want to remove ALL rows of those properties from pids_to_remove_f2.

# --- 5) Identify which geo_id values matched the MLS data ---
matched_geo_ids = set(merged["geo_id"])

# --- 6) Filter pids_to_remove_f2 to EXCLUDE any property that matched ---
pids_to_remove_f3 = df_more_than_onef[~df_more_than_onef["geo_id"].isin(matched_geo_ids)]

# Optionally, save or use this new DataFrame:
# pids_to_remove_f3.to_csv("Data/pids_to_remove_f3.csv", index=False)

print("Original pids_to_remove_f2 shape:", df_more_than_onef.shape)
print("Filtered pids_to_remove_f3 shape:", pids_to_remove_f3.shape)

pids_to_remove_f3.to_csv('Data/Outliers_to_review.csv')
# %%
import pandas as pd

# --- 1) Load the 2024 MLS Data Only ---
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')

# Add a "SaleYear" column if needed:
MLS_SalesXXIV["SaleYear"] = 2024

# If you already have df_more_than_onef loaded, skip re-reading it here
# df_more_than_onef = pd.read_csv('Data/df_more_than_onef.csv')

# --- 2) Compute the "sale year" we want from df_more_than_onef (prop_val_yr - 1) ---
df_more_than_onef["SaleYearWanted"] = df_more_than_onef["prop_val_yr"] - 1

# --- 3) Merge df_more_than_onef with **only the 2024 MLS** to see which properties match ---
# We'll match on:
#   - "geo_id" in df_more_than_onef
#   - "Tax ID" in MLS_SalesXXIV
#   - "SaleYearWanted" in df_more_than_onef == "SaleYear" in MLS_SalesXXIV (which is 2024)

merged_2024 = df_more_than_onef.merge(
    MLS_SalesXXIV,
    how="inner",
    left_on=["geo_id", "SaleYearWanted"],
    right_on=["Tax ID", "SaleYear"]
)

# 'merged_2024' now contains rows for any property-year combos that match 2024 specifically.

# --- 4) Identify which geo_id values matched the 2024 MLS data ---
matched_2024_geo_ids = set(merged_2024["geo_id"])

# --- 5) Filter df_more_than_onef to EXCLUDE any property that matched a 2024 sale ---
df_more_than_onef_f3 = df_more_than_onef[~df_more_than_onef["geo_id"].isin(matched_2024_geo_ids)]

# --- 6) Save or examine the results ---
print("Original df_more_than_onef shape:", df_more_than_onef.shape)
print("Filtered df_more_than_onef_f3 shape:", df_more_than_onef_f3.shape)

df_more_than_onef_f3.to_csv('Data/Outliers_to_review_2024only.csv', index=False)
# %%
# 2) Create a set of outlier IDs for fast membership checks
outlier_ids = set(hand_picked_outliers["prop_id"])

# 3) Filter df_more_than_onef_f3 so that we KEEP only rows whose prop_id is in hand_picked_outliers
df_more_than_onef_f4 = df_more_than_onef_f3[df_more_than_onef_f3["prop_id"].isin(outlier_ids)]

# 4) (Optional) Save the filtered DataFrame to a CSV
df_more_than_onef_f4.to_csv("Data/filtered_outliers.csv", index=False)

# %%
