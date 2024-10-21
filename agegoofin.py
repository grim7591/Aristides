# %%
# Load the CSV file
data = pd.read_csv('MapData20.csv')

# Calculate the actual age of properties
data['actual_age'] = 2024 - data['actual_year_built']

# Create a filter for properties over 100 years old
over_100_years_old = data['actual_age'] > 100

# Create a new stratification column based on effective year built being greater than 1994
data['effective_year_built_post_1994'] = (data['effective_year_built'] > 1994).astype(int)

# Filter data for properties over 100 years old
data_over_100_years = data[over_100_years_old]

# Group by the new stratification column and calculate sale ratio statistics
strata_stats = data_over_100_years.groupby('effective_year_built_post_1994')['sale_ratio'].agg(['mean', 'count']).reset_index()

# Rename the stratification column for clarity
strata_stats['strata'] = strata_stats['effective_year_built_post_1994'].replace({0: 'Effective Year Built <= 1994', 1: 'Effective Year Built > 1994'})

# Drop the old column and reorder
strata_stats = strata_stats.drop(columns=['effective_year_built_post_1994'])[['strata', 'mean', 'count']]

# Display the results
print(strata_stats)
# %%
# Calculate the actual age of properties
data['actual_age'] = 2024 - data['actual_year_built']

# Create a new stratification column based on effective year built being greater than 1994
data['effective_year_built_post_1994'] = (data['effective_year_built'] > 1994).astype(int)

# Create age group column in 10-year increments
age_bins_10_years = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
age_labels_10_years = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
data['age_group_10_years'] = pd.cut(data['actual_age'], bins=age_bins_10_years, labels=age_labels_10_years)

# Group by the stratification column and age group column and calculate sale ratio statistics
strata_age_group_stats = data.groupby(['effective_year_built_post_1994', 'age_group_10_years'])['sale_ratio'].agg(['mean', 'count']).reset_index()

# Rename the stratification column for clarity
strata_age_group_stats['effective_year_built'] = strata_age_group_stats['effective_year_built_post_1994'].replace({0: 'Effective Year Built <= 1994', 1: 'Effective Year Built > 1994'})

# Drop the old column and reorder
strata_age_group_stats = strata_age_group_stats.drop(columns=['effective_year_built_post_1994'])[['effective_year_built', 'age_group_10_years', 'mean', 'count']]

# Display the results
print(strata_age_group_stats)

# %%
# Calculate the actual age of properties
data['actual_age'] = 2024 - data['actual_year_built']

# Create a new stratification column based on effective year built being greater than 1994
data['effective_year_built_post_1994'] = (data['effective_year_built'] > 1994).astype(int)

# Create age group column in 10-year increments
age_bins_10_years = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
age_labels_10_years = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
data['age_group_10_years'] = pd.cut(data['actual_age'], bins=age_bins_10_years, labels=age_labels_10_years)

# Filter data for properties with effective year built <= 1994
data_effective_before_1994 = data[data['effective_year_built'] <= 1994]

# Group by age group and effective year built to get counts
strata_counts = data_effective_before_1994.groupby(['age_group_10_years', 'effective_year_built'])['sale_ratio'].count().reset_index()

# Rename columns for clarity
strata_counts = strata_counts.rename(columns={'sale_ratio': 'count'})

# Filter out rows with a count of 0
strata_counts_non_zero = strata_counts[strata_counts['count'] > 0]

# Display the filtered results
print(strata_counts_non_zero)
# %%
# Calculate the actual age of properties
data['actual_age'] = 2024 - data['actual_year_built']

# Create a new stratification column based on effective year built being greater than 1994
data['effective_year_built_post_1994'] = (data['effective_year_built'] > 1994).astype(int)

# Create age group column in 10-year increments
age_bins_10_years = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
age_labels_10_years = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
data['age_group_10_years'] = pd.cut(data['actual_age'], bins=age_bins_10_years, labels=age_labels_10_years)

# Filter data for properties with effective year built <= 1994
data_effective_before_1994 = data[data['effective_year_built'] <= 1994]

# Group by age group and effective year built to get both count and mean sale ratio
strata_stats = data_effective_before_1994.groupby(['age_group_10_years', 'effective_year_built']).agg(
    count=('sale_ratio', 'count'),
    mean_sale_ratio=('sale_ratio', 'mean')
).reset_index()

# Filter out rows with a count of 0
strata_stats_non_zero = strata_stats[strata_stats['count'] > 0]

# Display the filtered results with sale ratios
print(strata_stats_non_zero)

# %%
# Calculate the actual age of properties
data['actual_age'] = 2024 - data['actual_year_built']

# Create a new stratification column based on effective year built being greater than 1994
data['effective_year_built_post_1994'] = (data['effective_year_built'] > 1994).astype(int)

# Create age group column in 10-year increments
age_bins_10_years = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
age_labels_10_years = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
data['age_group_10_years'] = pd.cut(data['actual_age'], bins=age_bins_10_years, labels=age_labels_10_years)

# Filter data for properties with effective year built <= 1994
data_effective_before_1994 = data[data['effective_year_built'] <= 1994]

# Filter data for the specified strata to get property IDs
filtered_strata = data_effective_before_1994[
    ((data_effective_before_1994['age_group_10_years'] == '100+') & (data_effective_before_1994['effective_year_built'] == 1984)) |
    ((data_effective_before_1994['age_group_10_years'] == '71-80') & (data_effective_before_1994['effective_year_built'] == 1990)) |
    ((data_effective_before_1994['age_group_10_years'] == '100+') & (data_effective_before_1994['effective_year_built'] == 1975)) |
    ((data_effective_before_1994['age_group_10_years'] == '81-90') & (data_effective_before_1994['effective_year_built'] == 1990))
]

# Extract the property IDs for the filtered strata
property_ids = filtered_strata['prop_id']

# Convert the property IDs to a list and display them
property_ids_list = property_ids.tolist()
print(property_ids_list)
# %%
