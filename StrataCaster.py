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
    # Create a dictionary of preset bins for specific factors
    preset_bins = {
        'Assessment_Val': (113195, 209869, 210000, 295636, 295835, 422355, 423131, 3962781, np.inf),
        'living_area': (0, 999, 1000, 1499, 1500, 1999, 2000, 2499, 2500, 2999, 3000, 3499, 3500, np.inf),
        # I could not find ranges for legal acreage in the DOR review but I've got it in here so I want to check it. I used ArcGIS to make quantiles
        'legal_acreage': (0, 0.19, 0.20, 0.34, 0.35, np.inf), 
        'effective_year_built': (1, 1969, 1970, 1979, 1980, 1989, 1990, 1999, 2000, 2009, 2010, np.inf) 
    }
    # Create a list to store results
    stratified_results = []
    
    if bins is None:
        if factor in preset_bins:
            bins = preset_bins[factor]
        else:
            bins = None
            
    if bins is not None:
        # If bins are provided, use them to create the groups
        data[factor] = pd.cut(data[factor], bins=bins, right=False)
        
    
    # Stratify by the specified factor or binned factor
    for factor_value, group in data.groupby(factor):
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
