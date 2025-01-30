import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set file paths
base = "C:/Thimgan Dropbox/Clients/IAAO/Standard on Ratio Study/"
template = "C:/Thimgan Dropbox/Clients/TEMPLATE/"

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
ratio_stats = data.groupby('SYEAR')['Total'].agg(['count', 'median', 'mean', 'min', 'max'])

# Calculate percentiles for Proxy
percentiles = np.percentile(data['Proxy'], [10, 16.67, 33.33, 50, 66.67, 83.33, 90])

# Create Proxy Binned variables
data['ProxyBinned2'] = np.where(data['Proxy'] <= percentiles[2], 1, 2)
data['ProxyBinned4'] = pd.cut(data['Proxy'], bins=[-np.inf, percentiles[1], percentiles[2], percentiles[3], percentiles[4]], labels=[1, 2, 3, 4])
data['ProxyBinned6'] = pd.cut(data['Proxy'], bins=[-np.inf, percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4], percentiles[5]], labels=[1, 2, 3, 4, 5, 6])
data['ProxyBinned10'] = pd.cut(data['Proxy'], bins=[-np.inf, percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4], percentiles[5], percentiles[6]], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Calculate median changes across strata
median_changes = data.groupby(['ProxyBinned2', 'ProxyBinned4', 'ProxyBinned6', 'ProxyBinned10'])['sale_ratio'].median()

# Plot median values by strata
median_changes.plot(kind='bar')
plt.title("Median Values by Proxy Binned")
plt.show()
