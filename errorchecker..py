import pandas as pd

# Assuming 'count' is a Series generated from value_counts, with Market_Cluster_ID as the index
# And 'counts' is a Series for the filtered sale_ratio counts, also indexed by Market_Cluster_ID
count = result['Market_Cluster_ID'].value_counts()
import pandas as pd

# Load the CSV file into a DataFrame (replace 'your_file.csv' with the actual filename)
df = pd.read_csv('Data/errors.csv')

# Filter the rows where sale_ratio is over 0.95 or under 0.75
filtered_df = df[(df['sale_ratio'] > 0.95) | (df['sale_ratio'] < 0.75)]

# Group by 'Market_Cluster_ID', count the occurrences, and sort in descending order
counts = filtered_df.groupby('Market_Cluster_ID').size().sort_values(ascending=False)

# Display the result
print(counts)


# Convert both Series into DataFrames for merging
count_df = count.rename('total').reset_index().rename(columns={'index': 'Market_Cluster_ID'})
counts_df = counts.rename('relevant').reset_index().rename(columns={'index': 'Market_Cluster_ID'})

# Merge the two DataFrames on 'Market_Cluster_ID'
merged_df = pd.merge(count_df, counts_df, on='Market_Cluster_ID', how='left')

# Fill any missing relevant counts with 0 (in case some clusters have no relevant sale_ratio counts)
merged_df['relevant'].fillna(0, inplace=True)

# Calculate the percentage of relevant counts out of total counts
merged_df['percent_relevant'] = (merged_df['relevant'] / merged_df['total']) * 100

# Display the result
print(merged_df[['Market_Cluster_ID', 'relevant', 'total', 'percent_relevant']])
