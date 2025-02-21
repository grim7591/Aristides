# %%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
data = MapData.copy()

# Create `SYEAR` from `pxfer_date`
data['SYEAR'] = data['prop_val_yr'] - 1 

# Compute median for `sale_ratio`
overall_median = data['sale_ratio'].median()

# Calculate `Proxy`, `Ln_Proxy`, and `pct_diff`
data['Proxy'] = 0.5 * data['sl_price'] + 0.5 * (data['predicted_Market_Val'] / overall_median)
data['Ln_Proxy'] = np.log(data['Proxy']) / 0.693
data['pct_diff'] = (data['sale_ratio'] - overall_median) / overall_median

# Run linear regression
X = sm.add_constant(data['Ln_Proxy'])
model = sm.OLS(data['pct_diff'], X).fit()
print(model.summary())

# Scatter plot
plt.scatter(data['Ln_Proxy'], data['pct_diff'])
plt.xlabel("Ln_Proxy")
plt.ylabel("pct_diff")
plt.title("Scatter Plot of Ln_Proxy and pct_diff")
plt.show() 

# Calculate ratio statistics by year
ratio_stats = data.groupby('SYEAR')['predicted_Market_Val'].agg(['count', 'median', 'mean', 'min', 'max'])

# Calculate percentiles for Proxy
percentiles = np.percentile(data['Proxy'], [10, 16.67, 33.33, 50, 66.67, 83.33, 90])

# Calculate percentiles
deciles = np.percentile(data['Proxy'], [10, 16.67, 20, 25, 30, 33.33, 40, 50, 60, 66.67, 70, 75, 80, 83.33, 90])

# Assign named breakpoints
Per10, Per16_7, Per20, Per25, Per30, Per33_3, Per40, Per50, Per60, Per66_7, Per70, Per75, Per80, Per83_3, Per90 = deciles

# Create Proxy Binned variables
#data['ProxyBinned2'] = np.where(data['Proxy'] <= percentiles[2], 1, 2)
#data['ProxyBinned4'] = pd.cut(data['Proxy'], bins=[-np.inf, percentiles[1], percentiles[2], percentiles[3], percentiles[4]], labels=[1, 2, 3, 4])
#data['ProxyBinned6'] = pd.cut(data['Proxy'], bins=[-np.inf, percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4], percentiles[5]], labels=[1, 2, 3, 4, 5, 6])
# ProxyBinned10: Uses 10th, 20th, 30th, ..., 90th percentiles
data['ProxyBinned10'] = pd.cut(data['Proxy'], bins=[-np.inf, Per10, Per20, Per30, Per40, Per50, Per60, Per70, Per80, Per90, np.inf], 
                               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Calculate median changes across strata
median_changes = data.groupby(['ProxyBinned10'])['sale_ratio'].median()

# Plot median values by strata
median_changes.plot(kind='bar')
plt.title("Median Values by Proxy Binned")
plt.show()
# %%
# Plot median values by strata with rotated x-axis labels
median_changes = data.groupby(['ProxyBinned10'])['sale_ratio'].median()

# Reset the index for easier plotting
median_changes = median_changes.reset_index()

# Loop through each binning type and plot separately
for binned_column in ['ProxyBinned10']:
    plt.figure(figsize=(10, 6))
    subset = median_changes.groupby(binned_column)['sale_ratio'].median()
    subset.plot(kind='bar')
    plt.title(f"Median Values by {binned_column}")
    plt.ylabel("Median Sale Ratio")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to avoid clipping labels
    plt.show()
# %%
# Compute the median changes grouped by ProxyBinned2, ProxyBinned4, and ProxyBinned6
median_changes_table = data.groupby(['ProxyBinned10'])['sale_ratio'].median().reset_index()

# Rename columns for clarity
median_changes_table.columns = ['ProxyBinned10', 'Median_Sale_Ratio']

# Display the table
print(median_changes_table)

# %%
import numpy as np
import pandas as pd

# Function to compute confidence intervals for the median using bootstrapping
def median_confidence_interval(data, confidence=0.95, n_bootstrap=1000):
    """Compute the confidence interval for the median using bootstrapping."""
    bootstrapped_medians = np.median(np.random.choice(data, (n_bootstrap, len(data))), axis=1)
    lower_bound = np.percentile(bootstrapped_medians, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_medians, (1 + confidence) / 2 * 100)
    return lower_bound, upper_bound

# Compute median and confidence intervals for each bin
ci_results = []
for bin_label, group in data.groupby('ProxyBinned10'):
    median_value = group['sale_ratio'].median()
    lower_ci, upper_ci = median_confidence_interval(group['sale_ratio'].dropna().values)
    ci_results.append((bin_label, median_value, lower_ci, upper_ci))

# Create DataFrame
ci_df = pd.DataFrame(ci_results, columns=['ProxyBinned10', 'Median_Sale_Ratio', 'Lower_CI', 'Upper_CI'])

# Display results
print(ci_df)

# %%
# Function to calculate the median confidence interval using the prescribed method
def median_confidence_interval_exact(data, confidence=0.95):
    """Compute the confidence interval for the median using rank-based counting."""
    data = np.sort(data)  # Sort the data
    n = len(data)
    
    # Calculate j using the given formula
    j = (1.96 * np.sqrt(n)) / 2
    if n % 2 == 0:
        j += 0.5  # Adjust for even sample size
    j = int(np.ceil(j))  # Round up to the next largest integer

    # Find median
    median_value = np.median(data)
    
    # Determine lower and upper confidence limits
    lower_index = max(0, (n // 2) - j)  # Ensure index is not negative
    upper_index = min(n - 1, (n // 2) + j)  # Ensure index is within range
    lower_bound = data[lower_index]
    upper_bound = data[upper_index]

    return median_value, lower_bound, upper_bound

# Compute median and confidence intervals for each bin
ci_results = []
for bin_label, group in data.groupby('ProxyBinned10'):
    median_value, lower_ci, upper_ci = median_confidence_interval_exact(group['sale_ratio'].dropna().values)
    ci_results.append((bin_label, median_value, lower_ci, upper_ci))

# Create DataFrame
ci_df = pd.DataFrame(ci_results, columns=['ProxyBinned10', 'Median_Sale_Ratio', 'Lower_CI', 'Upper_CI'])

# Display results
print(ci_df)

# %%
