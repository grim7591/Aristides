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
from StrataAnalysisFunctions import StrataCaster

# Load the data
market_areas = pd.read_csv('Data/normalizedMAs.csv')
mass_data = pd.read_csv("Data/BeegData.csv")

# Clean the market area and sale data
market_areas = market_areas[['prop_id', 'MA', 'Cluster ID']]
market_areas.dropna(inplace=True)
market_areas = market_areas[market_areas['MA'] != '<Null>']
market_areas = market_areas[market_areas['prop_id'] != '<Null>']
market_areas['prop_id'] = market_areas['prop_id'].astype(str)

mass_data['prop_id'] = mass_data['prop_id'].astype(str)

# Factor engineer "Market Cluster ID"
market_areas['Market_Cluster_ID'] = market_areas['MA'].astype(str) + '_' + market_areas['Cluster ID'].astype(str)
market_areas['Market_Cluster_ID'] = market_areas['Market_Cluster_ID'].astype(str)

# Factor engineer "Assessment Val"
mass_data['Assessment_Val'] =.85 * (mass_data['market'] - (mass_data['Total_MISC_Val']/.85))

# Factor engineer "landiness"
avg_legal_acreage = (mass_data['legal_acreage']*43560).mean()
mass_data['landiness'] = (mass_data['legal_acreage']*43560) / avg_legal_acreage


# Merge the market area and sale data
result = pd.merge(mass_data, market_areas, how='inner', on='prop_id')
result.dropna(inplace=True)

# Make subdivision code a binary variable
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
result = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID', 'market', 'Total_MISC_Val'])

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
# %% Run the IAAO evaluation metrics on the test data
# Get predictions to test
predictions = result
predictions['predicted_log_Assessment_Val'] = regresult.predict(predictions)
predictions['predicted_Assessment_Val'] = np.exp(predictions['predicted_log_Assessment_Val'])
actual_values = result['Assessment_Val']
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
# %%
AVSA = StrataCaster(
    result,
    regresult,
    'Assessment_Val',
    None
)
LArSA = StrataCaster(
    result,
    regresult,
    'living_area',
    None
)
LAcSA = StrataCaster(
    result,
    regresult,
    'legal_acreage',
    None
)
QCSA = StrataCaster(
    result,
    regresult,
    'imprv_det_quality_cd',
    None
)
TASA = StrataCaster(
    result,
    regresult,
    'tax_area_description',
    None
)
MCSA = StrataCaster(
    result,
    regresult,
    'Market_Cluster_ID',
    None
)
# %%
AVSA.to_csv('Outputs/AVSA.csv', index=False)
LArSA.to_csv('Outputs/LArSA.csv', index=False)
LAcSA.to_csv('Outputs/LAcSA.csv', index=False)
QCSA.to_csv('Outputs/QCSA.csv', index=False)
TASA.to_csv('Outputs/TASA.csv', index=False)
MCSA.to_csv('Outputs/MCSA.csv', index=False)
# %%
StrataAnalysis = [AVSA, LArSA, LAcSA, QCSA, TASA, MCSA]
sheet_names = ['AVSA', 'LArSA', 'LAcSA', 'QCSA', 'TASA', 'MCSA']

# Create a Pandas Excel writer object
with pd.ExcelWriter('Outputs/StrataAnalysis.xlsx', engine='xlsxwriter') as writer:
    for StrataAnalysis, sheet in zip(StrataAnalysis, sheet_names):
        StrataAnalysis.to_excel(writer, sheet_name=sheet, index=False)
# %%
