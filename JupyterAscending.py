# %% Data preprocessing

# Import Libraries 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD, PRB, weightedMean, averageDeviation

# Load the data
market_areas = pd.read_csv('Data/normalizedMAs.csv')
sale_data = pd.read_csv("Data/dp22.csv")

# Clean the market area and sale data
market_areas = market_areas[['prop_id', 'MA', 'Cluster ID']]
market_areas.dropna(inplace=True)
market_areas = market_areas[market_areas['MA'] != '<Null>']
market_areas = market_areas[market_areas['prop_id'] != '<Null>']
market_areas['prop_id'] = market_areas['prop_id'].astype(str)

sale_data['prop_id'] = sale_data['prop_id'].astype(str)

# Factor engineer "Market Cluster ID"
market_areas['Market_Cluster_ID'] = market_areas['MA'].astype(str) + '_' + market_areas['Cluster ID'].astype(str)
market_areas['Market_Cluster_ID'] = market_areas['Market_Cluster_ID'].astype(str)

# Factor engineer "Assessment Val"
sale_data['Assessment_Val'] =.85 * (sale_data['sl_price'] - (sale_data['Total_MISC_Val']/.85))

# Factor engineer "landiness"
avg_legal_acreage = (sale_data['legal_acreage']*43560).mean()
sale_data['landiness'] = (sale_data['legal_acreage']*43560) / avg_legal_acreage
sale_data = sale_data.drop('legal_acreage', axis = 1)

# Merge the market area and sale data
result = pd.merge(sale_data, market_areas, how='inner', on='prop_id')
result.dropna(inplace=True)

# Make subdivision code a binary variable
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
result = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID', 'sl_price', 'Total_MISC_Val'])

# Factor Engineer Percent Good based on effective age
result['percent_good'] = 1- (result['effective_age']/100)
#result = result.drop(['effective_age'], axis =1)

# Linearize the quality codes
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    1: 0.75,
    2: 0.90,
    3: 1.00,
    4: 1.15,
    5: 1.40,
    6: 1.70
})

# Create dummy variables for non-numeric data, changing the name to data so I can use the un-dummied table later
result = result.join(pd.get_dummies(result.tax_area_description))
result = result.join(pd.get_dummies(result.Market_Cluster_ID))

# Rename columns that will act up in Python
column_mapping = {
    'HIGH SPRINGS' : 'HIGH_SPRINGS',
    "ST. JOHN'S" : 'ST_JOHNS',
    }
result.rename(columns=column_mapping, inplace=True)

# Ensure that all column names are strings
result.columns = result.columns.astype(str)
# %% Run some regression with logs in the formula
regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(landiness)+np.log(percent_good)+np.log(imprv_det_quality_cd)+np.log(total_porch_area+1)+np.log(total_garage_area+1)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision"
train_data, test_data = train_test_split(result, test_size=0.2, random_state=42)
regresult = smf.ols(formula=regressionFormula, data=train_data).fit()
regresult.summary()
# %% Run some regression with the values logged beforehand for testing/sanity purposes (commented out)
'''
result['legal_acreage'] = np.log(result['legal_acreage'])
result['Assessment_Val'] = np.log(result['Assessment_Val'])
result['living_area'] = np.log(result['living_area'])
result['percent_good'] = np.log(result['percent_good'])
result['total_porch_area'] = np.log(result['total_porch_area']+1)
result['total_garage_area'] = np.log(result['total_garage_area']+ 1)
result['imprv_det_quality_cd'] = np.log(result['imprv_det_quality_cd'])

regressionFormula_2 = """
Assessment_Val ~ living_area + legal_acreage + percent_good +
ALACHUA + ARCHER + GAINESVILLE + HAWTHORNE + HIGH_SPRINGS + NEWBERRY + 
WALDO + Springtree_B + HighSprings_A + MidtownEast_C + swNewberry_B + 
MidtownEast_A + swNewberry_A + MidtownEast_B + HighSprings_F + WaldoRural_C +
Springtree_A + Tioga_B + Tioga_A + swNewberry_C + MidtownEast_D + HighSprings_E +
MidtownEast_E + HighSprings_D + Springtree_C + WaldoRural_A + WaldoRural_B + 
HighSprings_C + MidtownEast_F + in_subdivision + imprv_det_quality_cd + total_porch_area + total_garage_area
"""
train_data, test_data = train_test_split(result, test_size=0.2, random_state=42)
regresult = smf.ols(formula=regressionFormula_2, data=train_data).fit()
regresult.summary()
'''
# %% Run the IAAO evaluation metrics on the test data
# Get predictions to test
predictions = test_data.copy()
predictions['predicted_log_Assessment_Val'] = regresult.predict(predictions)
predictions['predicted_Assessment_Val'] = np.exp(predictions['predicted_log_Assessment_Val'])
actual_values = predictions['Assessment_Val']
predicted_values = predictions['predicted_Assessment_Val']

# Test predictions on perfromance metrics
mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)
PRD_table = PRD(actual_values, predicted_values)
COD_table = COD(actual_values, predicted_values)
PRB_table = PRB(actual_values, predicted_values)
wm = weightedMean(actual_values, predicted_values)
ad = averageDeviation(actual_values, predicted_values)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"weightedMean: {wm}")
print(f"averageDevitation: {ad}")
# %% Statification by tax area analysis
# Create a list to store results
stratified_ta_results = []

# Stratify by tax area (replace 'tax_area_column' with the actual column name)
for tax_area, group in result.groupby('tax_area_description'):
    group_predictions = group.copy()
    group_predictions['predicted_log_Assessment_Val'] = regresult.predict(group)
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
    
    stratified_ta_results.append({
        'tax_area': tax_area,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'PRD': PRD_table,
        'COD': COD_table,
        'PRB': PRB_table,
        'Weighted Mean': wm,
        'Average Deviation': ad
    })
    count = group.shape[0]
# Convert the list of results to a DataFrame for easy viewing
stratified_ta_results_df = pd.DataFrame(stratified_ta_results)
print(stratified_ta_results_df)
# %% Statification by market area analysis
# Create a list to store results
stratified_ma_results = []

# Stratify by tax area (replace 'tax_area_column' with the actual column name)
for market_area, group in result.groupby('Market_Cluster_ID'):
    group_predictions = group.copy()
    group_predictions['predicted_log_Assessment_Val'] = regresult.predict(group)
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
    
    stratified_ma_results.append({
        'market area': market_area,
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
stratified_ma_results_df = pd.DataFrame(stratified_ma_results)
print(stratified_ma_results_df)
# %% Stratify by quality code analysis
# Create a list to store results
stratified_qc_results = []

for quality_code, group in result.groupby('imprv_det_quality_cd'):
    group_predictions = group.copy()
    group_predictions['predicted_log_Assessment_Val'] = regresult.predict(group)
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
    
    stratified_qc_results.append({
        'Quality': quality_code,
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
stratified_qc_results_df = pd.DataFrame(stratified_qc_results)
print(stratified_qc_results_df)
# %%