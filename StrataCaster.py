from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD, PRB, weightedMean, averageDeviation
import pandas as pd
import numpy as np

def StrataCaster(data, regression_result, factor, bins=None):
    """
    Perform stratification on the given data by the specified factor and calculate metrics.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the data.
    - regression_result: The fitted regression model.
    - factor (str): The column name by which to stratify the data.
    - factor_name (str, optional): A name for the factor to use in the output. Defaults to the column name.
    - bins (list or int, optional): The bins to use for grouping if the factor is continuous.

    Returns:
    - pd.DataFrame: A DataFrame containing the stratified results and metrics.
    """
    # Create a list to store results
    stratified_results = []

    if bins is not None:
        # If bins are provided, use them to create the groups
        data['binned_factor'] = pd.cut(data[factor], bins=bins)
        groupby_factor = 'binned_factor'
    else:
        groupby_factor = factor
    
    # Stratify by the specified factor or binned factor
    for factor_value, group in data.groupby(groupby_factor):
        group_predictions = group.copy()
        group_predictions['predicted_log_Assessment_Val'] = regression_result.predict(group)
        group_predictions['predicted_Assessment_Val'] = np.exp(group_predictions['predicted_log_Assessment_Val'])
        
        # Calculate metrics for the group
        actual_values = group_predictions['Assessment_Val']
        predicted_values = group_predictions['predicted_Assessment_Val']
        
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        PRD_table = PRD(actual_values, predicted_values)
        COD_table = COD(actual_values, predicted_values)
        PRB_table = PRB(actual_values, predicted_values)
        wm = weightedMean(actual_values, predicted_values)
        ad = averageDeviation(actual_values, predicted_values)
        
        count = group.shape[0]
        
        stratified_results.append({
            factor: factor_value,
            'Count': count,
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'PRD': PRD_table,
            'COD': COD_table,
            'PRB': PRB_table,
            'Weighted Mean': wm,
            'Average Deviation': ad
        })
    
    # Convert the list of results to a DataFrame for easy viewing
    stratified_results_df = pd.DataFrame(stratified_results)
    print(stratified_results_df)
    return stratified_results_df

# Example usage:
# For factors with defined values
# tax_area_results = StrataCaster(data, regression_result, 'tax_area_description', 'Tax Area')

# For continuous factors with ranges
# quality_code_results = StrataCaster(data, regression_result, 'imprv_det_quality_cd', 'Quality Code', bins=[1, 500, 1000])
