# %%
# Import Libraries 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD, PRB, weightedMean, averageDeviation, PRBCI
from StrataCaster import StrataCaster
from PlotPlotter import PlotPlotter
import matplotlib.pyplot as plt
from IPython.display import Markdown

# Load the data
print("Loading data from CSV file...")
result = pd.read_csv('Data/dp72m.csv')
result.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)

result['prop_id'] = result['prop_id'].astype(str)
result['geo_id'] = result['geo_id'].astype(str)

MLS_SalesXXI = pd.read_csv('Data/MLSData/2021MLSData.csv')
MLS_SalesXXII = pd.read_csv('Data/MLSData/2022MLSData.csv')
MLS_SalesXXIII = pd.read_csv('Data/MLSData/2023MLSData.csv')
MLS_SalesXXIV = pd.read_csv('Data/MLSData/2024MLSData.csv')

# 3) Convert 'Tax ID' columns to sets of strings for fast membership checking
mls_2021_set = set(MLS_SalesXXI['Tax ID'].astype(str))
mls_2022_set = set(MLS_SalesXXII['Tax ID'].astype(str))
mls_2023_set = set(MLS_SalesXXIII['Tax ID'].astype(str))
mls_2024_set = set(MLS_SalesXXIV['Tax ID'].astype(str))

# 4) Define the mapping: sale_year -> MLS set
#    The sale_year is prop_val_yr - 1. 
#    e.g. if prop_val_yr=2022 => sale_year=2021 => check mls_2021_set
sale_year_to_mls = {
    2021: mls_2021_set,
    2022: mls_2022_set,
    2023: mls_2023_set,
    2024: mls_2024_set
}

# 5) Create a 'sale_year' column = prop_val_yr - 1
result['sale_year'] = result['prop_val_yr'] - 1

# 6) Define a function to decide whether to keep a row
def keep_row(row):
    # Always keep if sl_county_ratio_cd == 2
    if row['sl_county_ratio_cd'] == 2:
        return True
    
    # Otherwise, check if we have an MLS set for this sale_year
    sy = row['sale_year']
    if sy in sale_year_to_mls:
        # Keep if the geo_id is in the corresponding MLS set
        if row['geo_id'] in sale_year_to_mls[sy]:
            return True
    
    # If neither condition is true, exclude
    return False

# 7) Filter the result DataFrame
result = result[result.apply(keep_row, axis=1)]

TrainTestOutliers = pd.read_csv('3IQR.csv')
result = result[~result['prop_id'].isin(TrainTestOutliers['prop_id'])]

# ### Overwriting the sale price of some properties whose sales were miscoded in PACs

result.loc[result['prop_id'] == '84296', 'sl_price'] = 90000
result.loc[result['prop_id'] == '79157', 'sl_price'] = 300000
result.loc[result['prop_id'] == '93683', 'sl_price'] = 199800
result.loc[result['prop_id'] == '93443', 'sl_price'] = 132500

# Factor engineer "Assessment Val"
print("Factor engineering Assessment Val...")
# Calculate the 'Assessment_Val' based on the sale price and miscellaneous value
result['Assessment_Val'] = .85 * (result['sl_price'] - (result['MISC_Val'] / .85))
# Add a validation step to ensure 'Assessment_Val' is not negative
result['Assessment_Val'] = result['Assessment_Val'].apply(lambda x: x if x > 0 else np.nan)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (result['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
result['landiness'] = (result['legal_acreage'] * 43560) / avg_legal_acreage

# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
result = result.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
result['prop_id'] = result['prop_id'].astype(str)

result['effective_age'] = result['effective_age'].apply(lambda x: 30 if x > 30 else x)

# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
result['percent_good'] = 1 - (result['effective_age']/ 100)

result.loc[result['prop_id'].isin(['96615']), 'imprv_det_quality_cd'] = 1

result.loc[result['prop_id'].isin(['96411', '13894', '8894']), 'imprv_det_quality_cd'] = 2

result.loc[result['prop_id'].isin(['91562', '73909']), 'imprv_det_quality_cd'] = 3

result.loc[result['prop_id'].isin(['19165']), 'imprv_det_quality_cd'] = 4

# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    1: 0.1331291,
    2: 0.5665645,
    3: 1.0,
    4: 1.1624432,
    5: 1.4343298,
    6: 1.7062164
})

# Create dummy variables for non-numeric data
print("Creating dummy variables...")
# Join dummy variables for 'tax_area_description' and 'Market_Cluster_ID'
result = result.join(pd.get_dummies(result.tax_area_description))
result = result.join(pd.get_dummies(result.Market_Cluster_ID))
#result = result.join(pd.get_dummies(result.School_Combination))
# Rename columns that will act up in Python
print("Renaming columns with problematic characters...")
# Rename columns to avoid issues with special characters or spaces
column_mapping = {
    'HIGH SPRINGS': 'HIGH_SPRINGS',
    "ST. JOHN'S": 'ST_JOHNS'
}
result.rename(columns=column_mapping, inplace=True)

# Define the variable
legalAcreageMax = 1  # in acres

result = result[result['legal_acreage'] < legalAcreageMax]

# Ensure that all column names are strings
result.columns = result.columns.astype(str)

regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + number_of_baths + in_subdivision + C(Market_Cluster_ID)"

# Split data into training and test sets
print("Splitting data into training and test sets...")
test_size_var = 0.2
train_data, test_data = train_test_split(result, test_size=test_size_var, random_state=42)

# Fit the regression model
print("Fitting the regression model...")
regresult = smf.ols(formula=regressionFormula, data=train_data).fit()
# Display regression summary
print("Regression model summary:")
print(regresult.summary())

print("Evaluating model performance on test data...")
# Get predictions to test
predictions = test_data.copy()
# Predict log-transformed assessment values
predictions['predicted_log_Assessment_Val'] = regresult.predict(predictions)
# Convert predicted log values to original scale
predictions['predicted_Assessment_Val'] = np.exp(predictions['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values = predictions['sl_price']
predicted_values = predictions['predicted_Assessment_Val'] + predictions['MISC_Val']
predicted_values_mae = predictions['predicted_Assessment_Val']
actual_values_mae = predictions['Assessment_Val']

# Test predictions on performance metrics
print("Calculating performance metrics...")
mae = mean_absolute_error(predicted_values, actual_values)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
# Calculate IAAO metrics
PRD_table = PRD(predicted_values, actual_values)
COD_table = COD(predicted_values, actual_values)
PRB_table = PRB(predicted_values, actual_values)
PRBCI_table = PRBCI(predicted_values, actual_values)
wm = weightedMean(predicted_values, actual_values)
meanRatio = (predicted_values / actual_values).mean()
medianRatio = (predicted_values / actual_values).median()

# Print performance metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"PRBCI: {PRBCI_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")
# %%
print("Performing geospatial analysis...")
# Create a copy of the result data for geospatial analysis
MapData = result.copy()
# Predict log-transformed assessment values for MapData
MapData['predicted_log_Assessment_Val'] = regresult.predict(MapData)
# Convert predicted log values to original scale
MapData['predicted_Assessment_Val'] = np.exp(MapData['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
MapData['predicted_Market_Val'] = MapData['predicted_Assessment_Val'] + MapData['MISC_Val']
# Calculate residuals for market and assessment values
MapData['Market_Residual'] = MapData['predicted_Market_Val'] - MapData['sl_price']
MapData['Assessment_Residual'] = MapData['predicted_Assessment_Val'] - MapData['Assessment_Val']
# Convert residuals to numeric and handle errors
MapData['Market_Residual'] = pd.to_numeric(MapData['Market_Residual'], errors='coerce')
MapData['Assessment_Residual'] = pd.to_numeric(MapData['Assessment_Residual'], errors='coerce')
# Calculate absolute values of residuals
MapData['AbsV_Market_Residual'] = MapData['Market_Residual'].abs()
MapData['AbsV_Assessment_Residual'] = MapData['Assessment_Residual'].abs()
# Calculate sale ratio
MapData['sale_ratio'] = MapData['predicted_Market_Val'] / MapData['sl_price']
# Export MapData to CSV
print("Exporting geospatial analysis data to CSV...")
MapData.to_csv('MapData.csv', index=False)
# %%
Q1 = MapData['sale_ratio'].quantile(0.25)
Q3 = MapData['sale_ratio'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

# Filter data within the bounds
outliers_df = MapData[(MapData['sale_ratio'] < lower_bound) | (MapData['sale_ratio'] > upper_bound)]


print("Filtered DataFrame:")
print(outliers_df)
outliers_df.to_csv('3IQR.csv')
# %%
