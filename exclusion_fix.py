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
