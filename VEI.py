import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

def VEI(MapData, confidence=0.95, n_bootstrap=1000, show_plots=True):
    """
    Analyzes property data from a DataFrame, including:
      - Calculating sale ratio medians, proxy values, and differences
      - Running a linear regression of pct_diff on Ln_Proxy
      - Plotting scatter plots
      - Calculating and plotting median changes by binned Proxy values
      - Computing confidence intervals via bootstrapping and an exact rank-based method
    
    Parameters
    ----------
    MapData : pd.DataFrame
        The input DataFrame with columns:
        ['prop_val_yr', 'pxfer_date', 'sale_ratio', 'sl_price', 'predicted_Market_Val'].
    confidence : float, optional
        Confidence level for median CI (default is 0.95).
    n_bootstrap : int, optional
        Number of bootstrap iterations for the bootstrapped CI (default is 1000).
    show_plots : bool, optional
        If True, plots are displayed. If False, plots are not shown (default is True).

    Returns
    -------
    results : dict
        A dictionary with the following keys:
        - 'regression_summary': the model summary as a string
        - 'median_changes_table': a DataFrame of median sale_ratio by ProxyBinned10
        - 'bootstrap_ci': a DataFrame of bin, median, lower_CI, upper_CI (bootstrapped)
        - 'exact_ci': a DataFrame of bin, median, lower_CI, upper_CI (exact rank-based)

    """
    
    # Copy original data to avoid modifying in place
    data = MapData.copy()

    # Create SYEAR
    data['SYEAR'] = data['prop_val_yr'] - 1

    # Compute the overall median of sale_ratio
    overall_median = data['sale_ratio'].median()

    # Calculate Proxy, Ln_Proxy, pct_diff
    data['Proxy'] = 0.5 * data['sl_price'] + 0.5 * (data['predicted_Market_Val'] / overall_median)
    data['Ln_Proxy'] = np.log(data['Proxy']) / 0.693  # dividing by ln(2)
    data['pct_diff'] = (data['sale_ratio'] - overall_median) / overall_median

    # Run linear regression
    X = sm.add_constant(data['Ln_Proxy'])
    model = sm.OLS(data['pct_diff'], X).fit()
    regression_summary = model.summary().as_text()

    # Scatter plot of Ln_Proxy and pct_diff
    if show_plots:
        plt.figure()
        plt.scatter(data['Ln_Proxy'], data['pct_diff'])
        plt.xlabel("Ln_Proxy")
        plt.ylabel("pct_diff")
        plt.title("Scatter Plot of Ln_Proxy vs. pct_diff")
        plt.show()

    # Calculate ratio statistics by year
    ratio_stats = data.groupby('SYEAR')['predicted_Market_Val'].agg(['count', 'median', 'mean', 'min', 'max'])

    # Calculate deciles for Proxy
    deciles = np.percentile(data['Proxy'], [10, 20, 30, 40, 50, 60, 70, 80, 90])
    (Per10, Per20, Per30, Per40, Per50, Per60, Per70, Per80, Per90) = deciles

    # Create ProxyBinned10
    data['ProxyBinned10'] = pd.cut(
        data['Proxy'],
        bins=[-np.inf, Per10, Per20, Per30, Per40, Per50, Per60, Per70, Per80, Per90, np.inf],
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    # Calculate and plot median changes by ProxyBinned10
    median_changes_table = data.groupby('ProxyBinned10')['sale_ratio'].median().reset_index()
    median_changes_table.columns = ['ProxyBinned10', 'Median_Sale_Ratio']

    if show_plots:
        plt.figure()
        median_changes_table.set_index('ProxyBinned10')['Median_Sale_Ratio'].plot(kind='bar')
        plt.title("Median Values by Proxy Binned (ProxyBinned10)")
        plt.xlabel("Proxy Bins (1=lowest, 10=highest)")
        plt.ylabel("Median Sale Ratio")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # Function for bootstrap-based median confidence interval
    def median_confidence_interval_bootstrap(data_values, confidence=0.95, n_bootstrap=1000):
        """Compute the confidence interval for the median using bootstrapping."""
        bootstrapped_medians = np.median(
            np.random.choice(data_values, (n_bootstrap, len(data_values))),
            axis=1
        )
        lower_bound = np.percentile(bootstrapped_medians, (1 - confidence) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_medians, (1 + confidence) / 2 * 100)
        return lower_bound, upper_bound

    # Compute bootstrap-based median CIs for each bin
    boot_ci_results = []
    for bin_label, group_df in data.groupby('ProxyBinned10'):
        median_value = group_df['sale_ratio'].median()
        lower_ci, upper_ci = median_confidence_interval_bootstrap(
            group_df['sale_ratio'].dropna().values,
            confidence=confidence,
            n_bootstrap=n_bootstrap
        )
        boot_ci_results.append((bin_label, median_value, lower_ci, upper_ci))
    
    bootstrap_ci = pd.DataFrame(
        boot_ci_results,
        columns=['ProxyBinned10', 'Median_Sale_Ratio', 'Lower_CI', 'Upper_CI']
    )

    # Function for an exact rank-based median confidence interval
    def median_confidence_interval_exact(data_values, confidence=0.95):
        """Compute the CI for the median using rank-based counting."""
        data_values = np.sort(data_values)
        n = len(data_values)

        # For a 95% CI, the z-score ~ 1.96. Generalize for other confidence levels if needed:
        # z = stats.norm.ppf(0.5 + confidence / 2.0)
        z = 1.96  # fixed for 95% if you prefer the original formula
        j = (z * np.sqrt(n)) / 2.0
        
        # Adjust for even sample sizes
        if n % 2 == 0:
            j += 0.5
        
        j = int(np.ceil(j))
        
        median_value = np.median(data_values)
        lower_index = max(0, (n // 2) - j)
        upper_index = min(n - 1, (n // 2) + j)
        lower_bound = data_values[lower_index]
        upper_bound = data_values[upper_index]
        return median_value, lower_bound, upper_bound

    # Compute exact rank-based median CIs for each bin
    exact_ci_results = []
    for bin_label, group_df in data.groupby('ProxyBinned10'):
        values = group_df['sale_ratio'].dropna().values
        if len(values) == 0:
            # Handle empty groups if needed
            exact_ci_results.append((bin_label, np.nan, np.nan, np.nan))
            continue
        median_val, lower_ci, upper_ci = median_confidence_interval_exact(values, confidence)
        exact_ci_results.append((bin_label, median_val, lower_ci, upper_ci))

    exact_ci = pd.DataFrame(
        exact_ci_results,
        columns=['ProxyBinned10', 'Median_Sale_Ratio', 'Lower_CI', 'Upper_CI']
    )

    # Compile results in a dictionary
    results = {
        "regression_summary": regression_summary,
        "median_changes_table": median_changes_table,
        "bootstrap_ci": bootstrap_ci,
        "exact_ci": exact_ci
    }
    
    return results
