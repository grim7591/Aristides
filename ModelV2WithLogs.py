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
from StrataCaster import StrataCaster
from PlotPlotter import PlotPlotter

# Load the data
print("Loading data from CSV files...")
# Load data from multiple CSV files
market_areas = pd.read_csv('Data/normalizedMAs.csv')
sale_data = pd.read_csv("Data/dp29.csv")

Haile = pd.read_csv("Data/Haile.csv")
High_Springs_Main = pd.read_csv("Data/High_Springs_Main.csv")
Turkey_Creek = pd.read_csv("Data/Turkey_Creek.csv")
Alachua_Main = pd.read_csv("Data/Alachua_Main.csv")
#Rural_UI = pd.read_csv("Data/Rural_UI.csv")
West_Outer_Gainesville = pd.read_csv("Data/West_Outer_Gainesville.csv")
Gainesvilleish_Region = pd.read_csv("Data/Gainesvilleish_Region.csv")
West_of_Waldo_rd = pd.read_csv("Data/West_of_Waldo_rd.csv")
Real_Tioga = pd.read_csv("Data/Real_Tioga.csv")
Duck_Pond = pd.read_csv("Data/DuckPond.csv")
Newmans_Lake = pd.read_csv("Data/Newmans_Lake.csv")
EastMidtownEastA = pd.read_csv("Data/EastMidtownEastA.csv")
HighSpringsAGNV = pd.read_csv("Data/HighSpringsAGNV.csv")
Thornebrooke = pd.read_csv("Data/Thornebrooke.csv")
HSBUI = pd.read_csv("Data/HSBUI.csv")
Golfview = pd.read_csv("Data/Golfview.csv")
Lugano = pd.read_csv("Data/Lugano.csv")
Archer = pd.read_csv("Data/Archer.csv")
WildsPlantation = pd.read_csv("Data/WildsPlantation.csv")

# Clean the market area and sale data
print("Cleaning market area and sale data...")
# Select only relevant columns from market_areas
market_areas = market_areas[['prop_id', 'MA', 'Cluster ID', 'CENTROID_X', 'CENTROID_Y', 'geo_id']]
# Remove rows with missing values
market_areas.dropna(inplace=True)
# Filter out rows with '<Null>' values
market_areas = market_areas[market_areas['MA'] != '<Null>']
market_areas = market_areas[market_areas['prop_id'] != '<Null>']
# Convert 'prop_id' to string type
market_areas['prop_id'] = market_areas['prop_id'].astype(str)

# Convert 'prop_id' in sale_data to string type
sale_data['prop_id'] = sale_data['prop_id'].astype(str)

# Factor engineer "Market Cluster ID"
print("Creating Market Cluster ID...")
# Create a new column 'Market_Cluster_ID' by combining 'MA' and 'Cluster ID'
market_areas['Market_Cluster_ID'] = market_areas['MA'].astype(str) + '_' + market_areas['Cluster ID'].astype(str)

# Factor engineer "Assessment Val"
print("Factor engineering Assessment Val...")
# Calculate the 'Assessment_Val' based on the sale price and miscellaneous value
sale_data['Assessment_Val'] = .85 * (sale_data['sl_price'] - (sale_data['Total_MISC_Val'] / .85))
# Add a validation step to ensure 'Assessment_Val' is not negative
sale_data['Assessment_Val'] = sale_data['Assessment_Val'].apply(lambda x: x if x > 0 else np.nan)

# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (sale_data['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
sale_data['landiness'] = (sale_data['legal_acreage'] * 43560) / avg_legal_acreage

# Merge the market area and sale data
print("Merging market area and sale data...")
# Merge sale_data and market_areas on 'prop_id'
result = pd.merge(sale_data, market_areas, how='inner', on='prop_id')
# Drop rows with missing values after merging
result.dropna(inplace=True)

# Make subdivision code and townhousery binary variables
print("Creating binary variables for subdivision and townhouse status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
result = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID'])
# Create a binary variable 'is_townhouse' to indicate if property is a townhouse
result['is_townhouse'] = result['imprv_type_cd'].apply(lambda x: True if x == '300' else False)
# Create a binary variable 'is_tiny' to indicate if living area is less than 1000 sq ft
result['is_tiny'] = result['living_area'].apply(lambda x: True if x < 1000 else False)

# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
result['percent_good'] = 1 - (result['effective_age'] / 100)

# Linearize the quality codes
print("Linearizing quality codes...")
# Replace quality codes with numerical values for linear regression
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    1: 0.75,
    2: 0.90,
    3: 1.00,
    4: 1.15,
    5: 1.40,
    6: 1.70
})

# New Market Area subdivisions
print("Updating Market Cluster IDs for new subdivisions...")
# Convert 'prop_id' to string for consistency across dataframes
result['prop_id'] = result['prop_id'].astype(str)

# Ensure 'prop_id' is a string for all subdivision dataframes
Haile['prop_id'] = Haile['prop_id'].astype(str)
High_Springs_Main['prop_id'] = High_Springs_Main['prop_id'].astype(str)
Turkey_Creek['prop_id'] = Turkey_Creek['prop_id'].astype(str)
Alachua_Main['prop_id'] = Alachua_Main['prop_id'].astype(str)
West_Outer_Gainesville['prop_id'] = West_Outer_Gainesville['prop_id'].astype(str)
Gainesvilleish_Region['prop_id'] = Gainesvilleish_Region['prop_id'].astype(str)
West_of_Waldo_rd['prop_id'] = West_of_Waldo_rd['prop_id'].astype(str)
Real_Tioga['prop_id'] = Real_Tioga['prop_id'].astype(str)
Duck_Pond['prop_id'] = Duck_Pond['prop_id'].astype(str)
Newmans_Lake['prop_id'] = Newmans_Lake['prop_id'].astype(str)
EastMidtownEastA['prop_id'] = EastMidtownEastA['prop_id'].astype(str)
HighSpringsAGNV['prop_id'] = HighSpringsAGNV['prop_id'].astype(str)
Thornebrooke['prop_id'] = Thornebrooke['prop_id'].astype(str)
HSBUI['prop_id'] = HSBUI['prop_id'].astype(str)
Golfview['prop_id'] = Golfview['prop_id'].astype(str)
Lugano['prop_id'] = Lugano['prop_id'].astype(str)
Archer['prop_id'] = Archer['prop_id'].astype(str)
WildsPlantation['prop_id'] = WildsPlantation['prop_id'].astype(str)

# Assign new Market Cluster IDs based on subdivision membership and tax area description
result.loc[result['prop_id'].isin(Haile['prop_id']), 'Market_Cluster_ID'] = 'Haile'
result.loc[result['Market_Cluster_ID'] == 'WaldoRural_B', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['tax_area_description'] == 'LACROSSE', 'Market_Cluster_ID'] = 'Lacrosse'
result.loc[result['tax_area_description'] == 'HAWTHORNE', 'Market_Cluster_ID'] = 'Hawthorne'
result.loc[result['Market_Cluster_ID'] == 'HighSprings_D', 'Market_Cluster_ID'] = 'High_Springs_Main'
result.loc[result['Market_Cluster_ID'] == 'MidtownEast_E', 'Market_Cluster_ID'] = 'MidtownEast_C'
result.loc[result['Market_Cluster_ID'] == 'MidtownEast_F', 'Market_Cluster_ID'] = 'MidtownEast_B'
result.loc[result['Market_Cluster_ID'] == 'HighSprings_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['Market_Cluster_ID'] == 'Springtree_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['Market_Cluster_ID'] == 'swNewberry_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['Market_Cluster_ID'] == 'WaldoRural_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['prop_id'].isin(High_Springs_Main['prop_id']), 'Market_Cluster_ID'] = 'High_Springs_Main'
result.loc[result['prop_id'].isin(Turkey_Creek['prop_id']), 'Market_Cluster_ID'] = 'Turkey_Creek'
result.loc[result['prop_id'].isin(Alachua_Main['prop_id']), 'Market_Cluster_ID'] = 'Alachua_Main'
result.loc[result['prop_id'].isin(West_Outer_Gainesville['prop_id']), 'Market_Cluster_ID'] = 'West_Outer_Gainesville'
result.loc[result['prop_id'].isin(Gainesvilleish_Region['prop_id']), 'Market_Cluster_ID'] = 'Gainesvilleish_Region'
result.loc[result['prop_id'].isin(West_of_Waldo_rd['prop_id']), 'Market_Cluster_ID'] = 'West_of_Waldo_rd'
result.loc[result['prop_id'].isin(Real_Tioga['prop_id']), 'Market_Cluster_ID'] = 'Real_Tioga'
result.loc[result['prop_id'].isin(Duck_Pond['prop_id']), 'Market_Cluster_ID'] = 'Duck_Pond'
result.loc[result['prop_id'].isin(Newmans_Lake['prop_id']), 'Market_Cluster_ID'] = 'Newmans_Lake'
result.loc[result['prop_id'].isin(EastMidtownEastA['prop_id']), 'Market_Cluster_ID'] = 'EastMidtownEastA'
result.loc[result['prop_id'].isin(HighSpringsAGNV['prop_id']), 'Market_Cluster_ID'] = 'HighSpringsAGNV'
result.loc[result['prop_id'].isin(Thornebrooke['prop_id']), 'Market_Cluster_ID'] = 'Thornebrooke'
result.loc[result['prop_id'].isin(HSBUI['prop_id']), 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['prop_id'].isin(Golfview['prop_id']), 'Market_Cluster_ID'] = 'Golfview'
result.loc[result['prop_id'].isin(Lugano['prop_id']), 'Market_Cluster_ID'] = 'Lugano'
result.loc[result['prop_id'].isin(Archer['prop_id']), 'Market_Cluster_ID'] = 'Archer'
result.loc[result['prop_id'].isin(WildsPlantation['prop_id']), 'Market_Cluster_ID'] = 'WildsPlantation'

# Create dummy variables for non-numeric data
print("Creating dummy variables...")
# Join dummy variables for 'tax_area_description' and 'Market_Cluster_ID'
result = result.join(pd.get_dummies(result.tax_area_description))
result = result.join(pd.get_dummies(result.Market_Cluster_ID))

# Rename columns that will act up in Python
print("Renaming columns with problematic characters...")
# Rename columns to avoid issues with special characters or spaces
column_mapping = {
    'HIGH SPRINGS': 'HIGH_SPRINGS',
    "ST. JOHN'S": 'ST_JOHNS'
}
result.rename(columns=column_mapping, inplace=True)

# Ensure that all column names are strings
result.columns = result.columns.astype(str)

# %% Run some regression with logs in the formula
print("Running regression model...")
# Regression formula with tax areas and townhouse-related variables
regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + Springtree_B + HighSprings_A + MidtownEast_C + swNewberry_B + MidtownEast_A + swNewberry_A + MidtownEast_B + HighSprings_F + Springtree_A + Tioga_B + Tioga_A + MidtownEast_D + HighSprings_E + WaldoRural_A + in_subdivision + West_Outer_Gainesville + Alachua_Main + High_Springs_Main + Haile + HighSprings_B + Lacrosse + West_of_Waldo_rd + Real_Tioga + Duck_Pond + Newmans_Lake + EastMidtownEastA + HighSpringsAGNV + Thornebrooke + Hawthorne + HSBUI + HighSprings_B + Golfview + Lugano + Archer + WildsPlantation"

# Split data into training and test sets
print("Splitting data into training and test sets...")
train_data, test_data = train_test_split(result, test_size=0.2, random_state=43)
# Fit the regression model
print("Fitting the regression model...")
regresult = smf.ols(formula=regressionFormula, data=train_data).fit()
# Display regression summary
print("Regression model summary:")
print(regresult.summary())

# %% Run the IAAO evaluation metrics on the test data
print("Evaluating model performance on test data...")
# Get predictions to test
predictions = test_data.copy()
# Predict log-transformed assessment values
predictions['predicted_log_Assessment_Val'] = regresult.predict(predictions)
# Convert predicted log values to original scale
predictions['predicted_Assessment_Val'] = np.exp(predictions['predicted_log_Assessment_Val'])
# Define actual and predicted values for further evaluation
actual_values = predictions['sl_price']
predicted_values = predictions['predicted_Assessment_Val'] + predictions['Total_MISC_Val']
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
wm = weightedMean(predicted_values, actual_values)
meanRatio = (predicted_values / actual_values).mean()
medianRatio = (predicted_values / actual_values).median()

# Print performance metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"weightedMean: {wm}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")

# %% Strata Analysis
print("Performing strata analysis...")
# Perform stratified analysis on different variables
AVSA = StrataCaster(result, regresult, 'Assessment_Val', None)
LArSA = StrataCaster(result, regresult, 'living_area', None)
LAcSA = StrataCaster(result, regresult, 'legal_acreage', None)
QCSA = StrataCaster(result, regresult, 'imprv_det_quality_cd', None)
TASA = StrataCaster(result, regresult, 'tax_area_description', None)
MCSA = StrataCaster(result, regresult, 'Market_Cluster_ID', None)

# %% Strata Analysis output
print("Exporting strata analysis results to CSV files...")
# Export the results of stratified analysis to CSV files
AVSA.to_csv('Outputs/AVSA.csv', index=False)
LArSA.to_csv('Outputs/LArSA.csv', index=False)
LAcSA.to_csv('Outputs/LAcSA.csv', index=False)
QCSA.to_csv('Outputs/QCSA.csv', index=False)
TASA.to_csv('Outputs/TASA.csv', index=False)
MCSA.to_csv('Outputs/MCSA.csv', index=False)

# %% Strata Analysis output part 2
print("Combining strata analysis results into an Excel file...")
# Combine stratified analysis results into a single Excel file with multiple sheets
StrataAnalysis = [AVSA, LArSA, LAcSA, QCSA, TASA, MCSA]
sheet_names = ['AVSA', 'LArSA', 'LAcSA', 'QCSA', 'TASA', 'MCSA']

# Create a Pandas Excel writer object
with pd.ExcelWriter('Outputs/StrataAnalysis.xlsx', engine='xlsxwriter') as writer:
    for analysis, sheet in zip(StrataAnalysis, sheet_names):
        analysis.to_excel(writer, sheet_name=sheet, index=False)

# %% Geospatial Analysis
print("Performing geospatial analysis...")
# Create a copy of the result data for geospatial analysis
MapData = result.copy()
# Predict log-transformed assessment values for MapData
MapData['predicted_log_Assessment_Val'] = regresult.predict(MapData)
# Convert predicted log values to original scale
MapData['predicted_Assessment_Val'] = np.exp(MapData['predicted_log_Assessment_Val'])
# Calculate predicted market value by adding miscellaneous value
MapData['predicted_Market_Val'] = MapData['predicted_Assessment_Val'] + MapData['Total_MISC_Val']
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

# %% Plot Analysis
print("Generating plots for different market clusters...")
# Generate plots for different market clusters
PlotPlotter(MapData, 'HighSprings_A')
PlotPlotter(MapData, 'HighSprings_B')
PlotPlotter(MapData, 'HighSprings_C')
PlotPlotter(MapData, 'HighSprings_D')
PlotPlotter(MapData, 'HighSprings_E')
PlotPlotter(MapData, 'HighSprings_F')
PlotPlotter(MapData, 'Springtree_A')
PlotPlotter(MapData, 'Springtree_B')
PlotPlotter(MapData, 'Springtree_C')
PlotPlotter(MapData, 'MidtownEast_A')
PlotPlotter(MapData, 'MidtownEast_B')
PlotPlotter(MapData, 'MidtownEast_C')
PlotPlotter(MapData, 'MidtownEast_D')
PlotPlotter(MapData, 'MidtownEast_E')
PlotPlotter(MapData, 'swNewberry_A')
PlotPlotter(MapData, 'swNewberry_B')
PlotPlotter(MapData, 'swNewberry_C')
PlotPlotter(MapData, 'WaldoRural_A')
PlotPlotter(MapData, 'WaldoRural_B')
PlotPlotter(MapData, 'WaldoRural_C')
PlotPlotter(MapData, 'Tioga_A')
PlotPlotter(MapData, 'Tioga_B')
PlotPlotter(MapData, '300')
# %%
