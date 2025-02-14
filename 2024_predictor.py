# %%
# 2024 only predictions
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD, PRB, weightedMean, averageDeviation, PRBCI

# PREDICTIONS NON MLS FILTERED
XXIVSales = pd.read_csv('Data/oopsall24sales_2m.csv')
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')
#XXIVOutliers = pd.read_csv('3IQRXXIV_3.csv')
#XXIVSales = XXIVSales[XXIVSales['geo_id'].isin(MLS_SalesXXIV['Tax ID'])]
#XXIVSales = XXIVSales[~XXIVSales['prop_id'].isin(XXIVOutliers['prop_id'])]
XXIVSales.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.tax_area_description))
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.Market_Cluster_ID))
XXIVSales.drop(XXIVSales[XXIVSales['legal_acreage'] >= 1].index, inplace=True)
#XXIVSales.drop(XXIVSales[XXIVSales['Market_Cluster_ID'] == 'Rural_North'].index, inplace=True)
XXIVSales.drop(XXIVSales[XXIVSales['Join_Count'] != 1].index, inplace=True)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (XXIVSales['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
XXIVSales['landiness'] = (XXIVSales['legal_acreage'] * 43560) / avg_legal_acreage
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
XXIVSales['in_subdivision'] = XXIVSales['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
XXIVSales = XXIVSales.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
XXIVSales['prop_id'] = XXIVSales['prop_id'].astype(str)

# ### Effective age overwrites
# In 2024 we updated the effective year built of all properties to 1994 at minimum. When reviewing outliers I applied that same logic to these properties which were evaluated on pre-2024 factors. It was determined by valuation that a 30 year limit on effective age makes sense and because that was a change in our process and not necessarily a market shift, I think it makes sense to mitigate the impact of that change on the model by applying it retroactively to previous sale years. 

XXIVSales['effective_age'] = XXIVSales['effective_age'].apply(lambda x: 30 if x > 30 else x)

# ### Calculating "percent good" from effective age
# Percent good = 1 - (effective_age/100)
# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
XXIVSales['percent_good'] = 1 - (XXIVSales['effective_age']/ 100)
# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
XXIVSales['imprv_det_quality_cd'] = XXIVSales['imprv_det_quality_cd'].replace({
    1: 0.1331291,
    2: 0.5665645,
    3: 1.0,
    4: 1.1624432,
    5: 1.4343298,
    6: 1.7062164
})

XXIVSales['predicted_log_Assessment_Val'] = regresult.predict(XXIVSales)

print("Evaluating model performance on test data...")
# Get predictions to test
predictions_2 = XXIVSales.copy()
# Predict log-transformed assessment values
#predictions_2['predicted_log_Assessment_Val'] = regresult.predict(predictions_2)
# Convert predicted log values to original scale
predictions_2['predicted_Assessment_Val'] = np.exp(predictions_2['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values_2 = predictions_2['sl_price']
predicted_values_2 = predictions_2['predicted_Assessment_Val'] + predictions_2['MISC_Val']
predicted_values_mae = predictions_2['predicted_Assessment_Val']
predictions_2['Assessment_Val'] = .85 * (predictions_2['sl_price'] - (predictions_2['MISC_Val'] / .85))
actual_values_mae = predictions_2['Assessment_Val']

# Test predictions on performance metrics
print("Calculating performance metrics...")
mae = mean_absolute_error(predicted_values_2, actual_values_2)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
# Calculate IAAO metrics
PRD_table = PRD(predicted_values_2, actual_values_2)
COD_table = COD(predicted_values_2, actual_values_2)
PRB_table = PRB(predicted_values_2, actual_values_2)
PRBCI_table = PRBCI(predicted_values_2, actual_values_2)
wm = weightedMean(predicted_values_2, actual_values_2)
meanRatio = (predicted_values_2 / actual_values_2).mean()
medianRatio = (predicted_values_2 / actual_values_2).median()

PRD_table_2 = PRD(predicted_values_mae, actual_values_mae)
COD_table_2 = COD(predicted_values_mae, actual_values_mae)
PRB_table_2 = PRB(predicted_values_mae, actual_values_mae)
PRBCI_table_2 = PRBCI(predicted_values_mae, actual_values_mae)
wm_2 = weightedMean(predicted_values_mae, actual_values_mae)
meanRatio_2 = (predicted_values_mae / actual_values_mae).mean()
medianRatio_2 = (predicted_values_mae / actual_values_mae).median()

# Print performance metrics
print("Non MLS filtered perfomance")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"PRBCI: {PRBCI_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")

''' print(f"PRD: {PRD_table_2}")
print(f"COD: {COD_table_2}")
print(f"PRB: {PRB_table_2}")
print(f"PRBCI: {PRBCI_table_2}")
print(f"weightedMean: {wm_2}")
print(f"meanRatio: {meanRatio_2}")
print(f"medianRatio: {medianRatio_2}") '''

# Convert predicted log values to original scale
XXIVSales['predicted_Assessment_Val'] = np.exp(XXIVSales['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
XXIVSales['predicted_Market_Val'] = XXIVSales['predicted_Assessment_Val'] + XXIVSales['MISC_Val']
# Calculate residuals for market and assessment values
XXIVSales['Market_Residual'] = XXIVSales['predicted_Market_Val'] - XXIVSales['sl_price']
#XXIVSales['Assessment_Residual'] = XXIVSales['predicted_Assessment_Val'] - XXIVSales['Assessment_Val']
# Convert residuals to numeric and handle errors
XXIVSales['Market_Residual'] = pd.to_numeric(XXIVSales['Market_Residual'], errors='coerce')
#XXIVSales['Assessment_Residual'] = pd.to_numeric(XXIVSales['Assessment_Residual'], errors='coerce')
# Calculate absolute values of residuals
XXIVSales['AbsV_Market_Residual'] = XXIVSales['Market_Residual'].abs()
#XXIVSales['AbsV_Assessment_Residual'] = XXIVSales['Assessment_Residual'].abs()
# Calculate sale ratio
XXIVSales['sale_ratio'] = XXIVSales['predicted_Market_Val'] / XXIVSales['sl_price']
# %%
# Calculate Q1, Q3, and IQR
Q1XXIV = XXIVSales['sale_ratio'].quantile(0.25)
Q3XXIV = XXIVSales['sale_ratio'].quantile(0.75)
IQRXXIV = Q3XXIV - Q1XXIV

# Define lower and upper bounds
lower_boundXXIV = Q1XXIV - 3 * IQRXXIV
upper_boundXXIV = Q3XXIV + 3 * IQRXXIV

# Filter data within the bounds
outliers_dfXXIV = XXIVSales[(XXIVSales['sale_ratio'] < lower_boundXXIV) | (XXIVSales['sale_ratio'] > upper_boundXXIV)]


print("Filtered DataFrame:")
print(outliers_dfXXIV)
outliers_dfXXIV.to_csv('3IQRXXIV_NoMLS.csv')

XXIVSales['market_ratio'] = XXIVSales['predicted_Market_Val'] /XXIVSales['sl_price']
BadBoySales = XXIVSales[
    (XXIVSales['market_ratio'].between(0.7, 1, inclusive='both')) &
    (~XXIVSales['sale_ratio'].between(0.7, 1, inclusive='both'))
]
BadBoySales.to_csv('Outputs/BadBoySales.csv')
# %%
# PREDICTIONS MLS FILTERED
XXIVSales = pd.read_csv('Data/oopsall24sales_2m.csv')
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')
#XXIVOutliers = pd.read_csv('3IQRXXIV_3.csv')
XXIVSales = XXIVSales[XXIVSales['geo_id'].isin(MLS_SalesXXIV['Tax ID'])]
#XXIVSales = XXIVSales[~XXIVSales['prop_id'].isin(XXIVOutliers['prop_id'])]
XXIVSales.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.tax_area_description))
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.Market_Cluster_ID))
XXIVSales.drop(XXIVSales[XXIVSales['legal_acreage'] >= 1].index, inplace=True)
#XXIVSales.drop(XXIVSales[XXIVSales['Market_Cluster_ID'] == 'Rural_North'].index, inplace=True)
XXIVSales.drop(XXIVSales[XXIVSales['Join_Count'] != 1].index, inplace=True)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (XXIVSales['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
XXIVSales['landiness'] = (XXIVSales['legal_acreage'] * 43560) / avg_legal_acreage
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
XXIVSales['in_subdivision'] = XXIVSales['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
XXIVSales = XXIVSales.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
XXIVSales['prop_id'] = XXIVSales['prop_id'].astype(str)

# ### Effective age overwrites
# In 2024 we updated the effective year built of all properties to 1994 at minimum. When reviewing outliers I applied that same logic to these properties which were evaluated on pre-2024 factors. It was determined by valuation that a 30 year limit on effective age makes sense and because that was a change in our process and not necessarily a market shift, I think it makes sense to mitigate the impact of that change on the model by applying it retroactively to previous sale years. 

XXIVSales['effective_age'] = XXIVSales['effective_age'].apply(lambda x: 30 if x > 30 else x)

# ### Calculating "percent good" from effective age
# Percent good = 1 - (effective_age/100)
# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
XXIVSales['percent_good'] = 1 - (XXIVSales['effective_age']/ 100)
# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
XXIVSales['imprv_det_quality_cd'] = XXIVSales['imprv_det_quality_cd'].replace({
    1: 0.1331291,
    2: 0.5665645,
    3: 1.0,
    4: 1.1624432,
    5: 1.4343298,
    6: 1.7062164
})

XXIVSales['predicted_log_Assessment_Val'] = regresult.predict(XXIVSales)

print("Evaluating model performance on test data...")
# Get predictions to test
predictions_2 = XXIVSales.copy()
# Predict log-transformed assessment values
#predictions_2['predicted_log_Assessment_Val'] = regresult.predict(predictions_2)
# Convert predicted log values to original scale
predictions_2['predicted_Assessment_Val'] = np.exp(predictions_2['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values_2 = predictions_2['sl_price']
predicted_values_2 = predictions_2['predicted_Assessment_Val'] + predictions_2['MISC_Val']
predicted_values_mae = predictions_2['predicted_Assessment_Val']
predictions_2['Assessment_Val'] = .85 * (predictions_2['sl_price'] - (predictions_2['MISC_Val'] / .85))
actual_values_mae = predictions_2['Assessment_Val']

# Test predictions on performance metrics
print("Calculating performance metrics...")
mae = mean_absolute_error(predicted_values_2, actual_values_2)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
# Calculate IAAO metrics
PRD_table = PRD(predicted_values_2, actual_values_2)
COD_table = COD(predicted_values_2, actual_values_2)
PRB_table = PRB(predicted_values_2, actual_values_2)
PRBCI_table = PRBCI(predicted_values_2, actual_values_2)
wm = weightedMean(predicted_values_2, actual_values_2)
meanRatio = (predicted_values_2 / actual_values_2).mean()
medianRatio = (predicted_values_2 / actual_values_2).median()

PRD_table_2 = PRD(predicted_values_mae, actual_values_mae)
COD_table_2 = COD(predicted_values_mae, actual_values_mae)
PRB_table_2 = PRB(predicted_values_mae, actual_values_mae)
PRBCI_table_2 = PRBCI(predicted_values_mae, actual_values_mae)
wm_2 = weightedMean(predicted_values_mae, actual_values_mae)
meanRatio_2 = (predicted_values_mae / actual_values_mae).mean()
medianRatio_2 = (predicted_values_mae / actual_values_mae).median()

# Print performance metrics
print("MLS filtered perfomance")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"PRBCI: {PRBCI_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")

''' print(f"PRD: {PRD_table_2}")
print(f"COD: {COD_table_2}")
print(f"PRB: {PRB_table_2}")
print(f"PRBCI: {PRBCI_table_2}")
print(f"weightedMean: {wm_2}")
print(f"meanRatio: {meanRatio_2}")
print(f"medianRatio: {medianRatio_2}") '''

# Convert predicted log values to original scale
XXIVSales['predicted_Assessment_Val'] = np.exp(XXIVSales['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
XXIVSales['predicted_Market_Val'] = XXIVSales['predicted_Assessment_Val'] + XXIVSales['MISC_Val']
# Calculate residuals for market and assessment values
XXIVSales['Market_Residual'] = XXIVSales['predicted_Market_Val'] - XXIVSales['sl_price']
#XXIVSales['Assessment_Residual'] = XXIVSales['predicted_Assessment_Val'] - XXIVSales['Assessment_Val']
# Convert residuals to numeric and handle errors
XXIVSales['Market_Residual'] = pd.to_numeric(XXIVSales['Market_Residual'], errors='coerce')
#XXIVSales['Assessment_Residual'] = pd.to_numeric(XXIVSales['Assessment_Residual'], errors='coerce')
# Calculate absolute values of residuals
XXIVSales['AbsV_Market_Residual'] = XXIVSales['Market_Residual'].abs()
#XXIVSales['AbsV_Assessment_Residual'] = XXIVSales['Assessment_Residual'].abs()
# Calculate sale ratio
XXIVSales['sale_ratio'] = XXIVSales['predicted_Market_Val'] / XXIVSales['sl_price']

# %%
# Calculate Q1, Q3, and IQR
Q1XXIV = XXIVSales['sale_ratio'].quantile(0.25)
Q3XXIV = XXIVSales['sale_ratio'].quantile(0.75)
IQRXXIV = Q3XXIV - Q1XXIV

# Define lower and upper bounds
lower_boundXXIV = Q1XXIV - 3 * IQRXXIV
upper_boundXXIV = Q3XXIV + 3 * IQRXXIV

# Filter data within the bounds
outliers_dfXXIV = XXIVSales[(XXIVSales['sale_ratio'] < lower_boundXXIV) | (XXIVSales['sale_ratio'] > upper_boundXXIV)]


print("Filtered DataFrame:")
print(outliers_dfXXIV)
outliers_dfXXIV.to_csv('3IQRXXIV_MLS.csv')
# %%
# PREDICTIONS  3 * IQR filtered
XXIVSales = pd.read_csv('Data/oopsall24sales_2m.csv')
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')
XXIVOutliers = pd.read_csv('3IQRXXIV_NoMLS.csv')
#XXIVSales = XXIVSales[XXIVSales['geo_id'].isin(MLS_SalesXXIV['Tax ID'])]
XXIVSales = XXIVSales[~XXIVSales['prop_id'].isin(XXIVOutliers['prop_id'])]
XXIVSales.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.tax_area_description))
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.Market_Cluster_ID))
XXIVSales.drop(XXIVSales[XXIVSales['legal_acreage'] >= 1].index, inplace=True)
#XXIVSales.drop(XXIVSales[XXIVSales['Market_Cluster_ID'] == 'Rural_North'].index, inplace=True)
XXIVSales.drop(XXIVSales[XXIVSales['Join_Count'] != 1].index, inplace=True)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (XXIVSales['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
XXIVSales['landiness'] = (XXIVSales['legal_acreage'] * 43560) / avg_legal_acreage
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
XXIVSales['in_subdivision'] = XXIVSales['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
XXIVSales = XXIVSales.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
XXIVSales['prop_id'] = XXIVSales['prop_id'].astype(str)

# ### Effective age overwrites
# In 2024 we updated the effective year built of all properties to 1994 at minimum. When reviewing outliers I applied that same logic to these properties which were evaluated on pre-2024 factors. It was determined by valuation that a 30 year limit on effective age makes sense and because that was a change in our process and not necessarily a market shift, I think it makes sense to mitigate the impact of that change on the model by applying it retroactively to previous sale years. 

XXIVSales['effective_age'] = XXIVSales['effective_age'].apply(lambda x: 30 if x > 30 else x)

# ### Calculating "percent good" from effective age
# Percent good = 1 - (effective_age/100)
# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
XXIVSales['percent_good'] = 1 - (XXIVSales['effective_age']/ 100)
# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
XXIVSales['imprv_det_quality_cd'] = XXIVSales['imprv_det_quality_cd'].replace({
    1: 0.1331291,
    2: 0.5665645,
    3: 1.0,
    4: 1.1624432,
    5: 1.4343298,
    6: 1.7062164
})

XXIVSales['predicted_log_Assessment_Val'] = regresult.predict(XXIVSales)

print("Evaluating model performance on test data...")
# Get predictions to test
predictions_2 = XXIVSales.copy()
# Predict log-transformed assessment values
#predictions_2['predicted_log_Assessment_Val'] = regresult.predict(predictions_2)
# Convert predicted log values to original scale
predictions_2['predicted_Assessment_Val'] = np.exp(predictions_2['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values_2 = predictions_2['sl_price']
predicted_values_2 = predictions_2['predicted_Assessment_Val'] + predictions_2['MISC_Val']
predicted_values_mae = predictions_2['predicted_Assessment_Val']
predictions_2['Assessment_Val'] = .85 * (predictions_2['sl_price'] - (predictions_2['MISC_Val'] / .85))
actual_values_mae = predictions_2['Assessment_Val']

# Test predictions on performance metrics
print("Calculating performance metrics...")
mae = mean_absolute_error(predicted_values_2, actual_values_2)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
# Calculate IAAO metrics
PRD_table = PRD(predicted_values_2, actual_values_2)
COD_table = COD(predicted_values_2, actual_values_2)
PRB_table = PRB(predicted_values_2, actual_values_2)
PRBCI_table = PRBCI(predicted_values_2, actual_values_2)
wm = weightedMean(predicted_values_2, actual_values_2)
meanRatio = (predicted_values_2 / actual_values_2).mean()
medianRatio = (predicted_values_2 / actual_values_2).median()

PRD_table_2 = PRD(predicted_values_mae, actual_values_mae)
COD_table_2 = COD(predicted_values_mae, actual_values_mae)
PRB_table_2 = PRB(predicted_values_mae, actual_values_mae)
PRBCI_table_2 = PRBCI(predicted_values_mae, actual_values_mae)
wm_2 = weightedMean(predicted_values_mae, actual_values_mae)
meanRatio_2 = (predicted_values_mae / actual_values_mae).mean()
medianRatio_2 = (predicted_values_mae / actual_values_mae).median()

# Print performance metrics
print("3*IQR filtered perfomance")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"PRBCI: {PRBCI_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")

''' print(f"PRD: {PRD_table_2}")
print(f"COD: {COD_table_2}")
print(f"PRB: {PRB_table_2}")
print(f"PRBCI: {PRBCI_table_2}")
print(f"weightedMean: {wm_2}")
print(f"meanRatio: {meanRatio_2}")
print(f"medianRatio: {medianRatio_2}") '''

# Convert predicted log values to original scale
XXIVSales['predicted_Assessment_Val'] = np.exp(XXIVSales['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
XXIVSales['predicted_Market_Val'] = XXIVSales['predicted_Assessment_Val'] + XXIVSales['MISC_Val']
# Calculate residuals for market and assessment values
XXIVSales['Market_Residual'] = XXIVSales['predicted_Market_Val'] - XXIVSales['sl_price']
#XXIVSales['Assessment_Residual'] = XXIVSales['predicted_Assessment_Val'] - XXIVSales['Assessment_Val']
# Convert residuals to numeric and handle errors
XXIVSales['Market_Residual'] = pd.to_numeric(XXIVSales['Market_Residual'], errors='coerce')
#XXIVSales['Assessment_Residual'] = pd.to_numeric(XXIVSales['Assessment_Residual'], errors='coerce')
# Calculate absolute values of residuals
XXIVSales['AbsV_Market_Residual'] = XXIVSales['Market_Residual'].abs()
#XXIVSales['AbsV_Assessment_Residual'] = XXIVSales['Assessment_Residual'].abs()
# Calculate sale ratio
XXIVSales['sale_ratio'] = XXIVSales['predicted_Market_Val'] / XXIVSales['sl_price']
# %%
# PREDICTIONS MLS + 3 * IQR filtered
XXIVSales = pd.read_csv('Data/oopsall24sales_2m.csv')
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')
XXIVOutliers = pd.read_csv('3IQRXXIV_MLS.csv')
XXIVSales = XXIVSales[XXIVSales['geo_id'].isin(MLS_SalesXXIV['Tax ID'])]
XXIVSales = XXIVSales[~XXIVSales['prop_id'].isin(XXIVOutliers['prop_id'])]
XXIVSales.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.tax_area_description))
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.Market_Cluster_ID))
XXIVSales.drop(XXIVSales[XXIVSales['legal_acreage'] >= 1].index, inplace=True)
#XXIVSales.drop(XXIVSales[XXIVSales['Market_Cluster_ID'] == 'Rural_North'].index, inplace=True)
XXIVSales.drop(XXIVSales[XXIVSales['Join_Count'] != 1].index, inplace=True)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (XXIVSales['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
XXIVSales['landiness'] = (XXIVSales['legal_acreage'] * 43560) / avg_legal_acreage
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
XXIVSales['in_subdivision'] = XXIVSales['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
XXIVSales = XXIVSales.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
XXIVSales['prop_id'] = XXIVSales['prop_id'].astype(str)

# ### Effective age overwrites
# In 2024 we updated the effective year built of all properties to 1994 at minimum. When reviewing outliers I applied that same logic to these properties which were evaluated on pre-2024 factors. It was determined by valuation that a 30 year limit on effective age makes sense and because that was a change in our process and not necessarily a market shift, I think it makes sense to mitigate the impact of that change on the model by applying it retroactively to previous sale years. 

XXIVSales['effective_age'] = XXIVSales['effective_age'].apply(lambda x: 30 if x > 30 else x)

# ### Calculating "percent good" from effective age
# Percent good = 1 - (effective_age/100)
# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
XXIVSales['percent_good'] = 1 - (XXIVSales['effective_age']/ 100)
# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
XXIVSales['imprv_det_quality_cd'] = XXIVSales['imprv_det_quality_cd'].replace({
    1: 0.1331291,
    2: 0.5665645,
    3: 1.0,
    4: 1.1624432,
    5: 1.4343298,
    6: 1.7062164
})

XXIVSales['predicted_log_Assessment_Val'] = regresult.predict(XXIVSales)

print("Evaluating model performance on test data...")
# Get predictions to test
predictions_2 = XXIVSales.copy()
# Predict log-transformed assessment values
#predictions_2['predicted_log_Assessment_Val'] = regresult.predict(predictions_2)
# Convert predicted log values to original scale
predictions_2['predicted_Assessment_Val'] = np.exp(predictions_2['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values_2 = predictions_2['sl_price']
predicted_values_2 = predictions_2['predicted_Assessment_Val'] + predictions_2['MISC_Val']
predicted_values_mae = predictions_2['predicted_Assessment_Val']
predictions_2['Assessment_Val'] = .85 * (predictions_2['sl_price'] - (predictions_2['MISC_Val'] / .85))
actual_values_mae = predictions_2['Assessment_Val']

# Test predictions on performance metrics
print("Calculating performance metrics...")
mae = mean_absolute_error(predicted_values_2, actual_values_2)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
# Calculate IAAO metrics
PRD_table = PRD(predicted_values_2, actual_values_2)
COD_table = COD(predicted_values_2, actual_values_2)
PRB_table = PRB(predicted_values_2, actual_values_2)
PRBCI_table = PRBCI(predicted_values_2, actual_values_2)
wm = weightedMean(predicted_values_2, actual_values_2)
meanRatio = (predicted_values_2 / actual_values_2).mean()
medianRatio = (predicted_values_2 / actual_values_2).median()

PRD_table_2 = PRD(predicted_values_mae, actual_values_mae)
COD_table_2 = COD(predicted_values_mae, actual_values_mae)
PRB_table_2 = PRB(predicted_values_mae, actual_values_mae)
PRBCI_table_2 = PRBCI(predicted_values_mae, actual_values_mae)
wm_2 = weightedMean(predicted_values_mae, actual_values_mae)
meanRatio_2 = (predicted_values_mae / actual_values_mae).mean()
medianRatio_2 = (predicted_values_mae / actual_values_mae).median()

# Print performance metrics
print("MLS + 3IQR filtered perfomance")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"PRBCI: {PRBCI_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")

''' print(f"PRD: {PRD_table_2}")
print(f"COD: {COD_table_2}")
print(f"PRB: {PRB_table_2}")
print(f"PRBCI: {PRBCI_table_2}")
print(f"weightedMean: {wm_2}")
print(f"meanRatio: {meanRatio_2}")
print(f"medianRatio: {medianRatio_2}") '''

# Convert predicted log values to original scale
XXIVSales['predicted_Assessment_Val'] = np.exp(XXIVSales['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
XXIVSales['predicted_Market_Val'] = XXIVSales['predicted_Assessment_Val'] + XXIVSales['MISC_Val']
# Calculate residuals for market and assessment values
XXIVSales['Market_Residual'] = XXIVSales['predicted_Market_Val'] - XXIVSales['sl_price']
#XXIVSales['Assessment_Residual'] = XXIVSales['predicted_Assessment_Val'] - XXIVSales['Assessment_Val']
# Convert residuals to numeric and handle errors
XXIVSales['Market_Residual'] = pd.to_numeric(XXIVSales['Market_Residual'], errors='coerce')
#XXIVSales['Assessment_Residual'] = pd.to_numeric(XXIVSales['Assessment_Residual'], errors='coerce')
# Calculate absolute values of residuals
XXIVSales['AbsV_Market_Residual'] = XXIVSales['Market_Residual'].abs()
#XXIVSales['AbsV_Assessment_Residual'] = XXIVSales['Assessment_Residual'].abs()
# Calculate sale ratio
XXIVSales['sale_ratio'] = XXIVSales['predicted_Market_Val'] / XXIVSales['sl_price']
# %%