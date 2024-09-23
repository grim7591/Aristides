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
market_areas = pd.read_csv('Data/normalizedMAs.csv')
sale_data = pd.read_csv("Data/dp26.csv")
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
SouthNewmansLake = pd.read_csv("Data/SouthNewmansLake.csv")
HighSpringsAGNV = pd.read_csv("Data/HighSpringsAGNV.csv")
Thornebrooke = pd.read_csv("Data/Thornebrooke.csv")

# Clean the market area and sale data
market_areas = market_areas[['prop_id', 'MA', 'Cluster ID', 'CENTROID_X', 'CENTROID_Y', 'geo_id']]
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


# Merge the market area and sale data
result = pd.merge(sale_data, market_areas, how='inner', on='prop_id')
result.dropna(inplace=True)


# Make subdivision code and townhousery binary variables
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
result = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID'])
result['is_townhouse'] = result['imprv_type_cd'].apply(lambda x: True if x == '300' else False)
result['is_tiny'] = result['living_area'].apply(lambda x: True if x < 1000 else False)

# Factor Engineer Percent Good based on effective age
result['percent_good'] = 1- (result['effective_age']/100)
#result = result.drop(['effective_age'], axis =1)
'''
# Market Area Adjustments
market_areas_2['prop_id'] = market_areas_2['prop_id'].astype(str)
result = pd.merge(result, market_areas_2, how='left', on='prop_id')

result['Market_Cluster_ID_2'] = result['Market_Cluster_ID']

result.loc[result['Cluster ID'] == 1, 'Market_Cluster_ID_2'] = 'HighSprings_B1'
result.loc[result['Cluster ID'] == 2, 'Market_Cluster_ID_2'] = 'HighSprings_B2'
'''
# Linearize the quality codes
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    1: 0.75,
    2: 0.90,
    3: 1.00,
    4: 1.15,
    5: 1.40,
    6: 1.70
})

# New Market Area subdivisions
result['prop_id'] = result['prop_id'].astype(str)
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
SouthNewmansLake['prop_id'] = SouthNewmansLake['prop_id'].astype(str)
HighSpringsAGNV['prop_id'] = HighSpringsAGNV['prop_id'].astype(str)
Thornebrooke['prop_id'] = Thornebrooke['prop_id'].astype(str)

#Rural_UI['prop_id'] = Rural_UI['prop_id'].astype(str)
result.loc[result['prop_id'].isin(Haile['prop_id']), 'Market_Cluster_ID'] = 'Haile'
result.loc[result['tax_area_description'] == 'LACROSSE', 'Market_Cluster_ID'] = 'Lacrosse'
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
result.loc[result['prop_id'].isin(SouthNewmansLake['prop_id']), 'Market_Cluster_ID'] = 'SouthNewmansLake'
result.loc[result['prop_id'].isin(HighSpringsAGNV['prop_id']), 'Market_Cluster_ID'] = 'HighSpringsAGNV'
result.loc[result['prop_id'].isin(Thornebrooke['prop_id']), 'Market_Cluster_ID'] = 'Thornebrooke'
#result.loc[result['prop_id'].isin(Rural_UI['prop_id']), 'Market_Cluster_ID'] = 'Rural_UI'

# Create dummy variables for non-numeric data, changing the name to data so I can use the un-dummied table later
result = result.join(pd.get_dummies(result.tax_area_description))
result = result.join(pd.get_dummies(result.Market_Cluster_ID))
#result = result.join(pd.get_dummies(result.imprv_type_cd))

# Rename columns that will act up in Python
column_mapping = {
    'HIGH SPRINGS' : 'HIGH_SPRINGS',
    "ST. JOHN'S" : 'ST_JOHNS',
    #'100' : 'SFH',
    #'200' : 'SFR - MFG',
    ##'300' : 'Zero_Lot'
    #''
    }
result.rename(columns=column_mapping, inplace=True)

# Ensure that all column names are strings
result.columns = result.columns.astype(str)
# %% Run some regression with logs in the formula
# Regression formula with tax areas and townhosue stuff

#regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(landiness)+np.log(percent_good)+np.log(imprv_det_quality_cd)+np.log(total_porch_area+1)+np.log(total_garage_area+1)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision+is_townhouse+np.log(1+(sum_us_area/living_area))+is_tiny"

# Regression formula without tax areas or any of the towhouse stuff.

#regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(landiness)+np.log(percent_good)+np.log(imprv_det_quality_cd)+np.log(total_porch_area+1)+np.log(total_garage_area+1)+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision+np.log(1+(sum_us_area/living_area))+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO"

# Current working model

#regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(landiness)+np.log(percent_good)+np.log(imprv_det_quality_cd)+np.log(total_porch_area+1)+np.log(total_garage_area+1)+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision"

# With new submarkets

regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(landiness)+np.log(percent_good)+np.log(imprv_det_quality_cd)+np.log(total_porch_area+1)+np.log(total_garage_area+1)+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision+West_Outer_Gainesville+Alachua_Main+High_Springs_Main+Haile+HighSprings_B+Lacrosse+West_of_Waldo_rd+Real_Tioga+Duck_Pond+Newmans_Lake+EastMidtownEastA+SouthNewmansLake+HighSpringsAGNV+Thornebrooke"

train_data, test_data = train_test_split(result, test_size=0.2, random_state=43)
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
actual_values = predictions['sl_price']
predicted_values = predictions['predicted_Assessment_Val'] + predictions ['Total_MISC_Val']
predicted_values_mae = predictions['predicted_Assessment_Val']
actual_values_mae = predictions['Assessment_Val']

# Test predictions on perfromance metrics
mae = mean_absolute_error(predicted_values, actual_values)
mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)
#mse = mean_squared_error(actual_values, predicted_values)
#r2 = r2_score(actual_values, predicted_values)
PRD_table = PRD(predicted_values,actual_values)
COD_table = COD(predicted_values,actual_values)
PRB_table = PRB(predicted_values,actual_values)
wm = weightedMean(predicted_values,actual_values)
#ad = averageDeviation(actual_values, predicted_values)
meanRatio = (predicted_values / actual_values).mean()
medianRatio = (predicted_values / actual_values).median()

print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Error_2: {mae_2}")
#print(f"Mean Squared Error: {mse}")
#print(f"R-squared: {r2}")
print(f"PRD: {PRD_table}")
print(f"COD: {COD_table}")
print(f"PRB: {PRB_table}")
print(f"weightedMean: {wm}")
#print(f"averageDevitation: {ad}")
print(f"meanRatio: {meanRatio}")
print(f"medianRatio: {medianRatio}")
# %% Strata Analysis
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
# %% Strata Analysis output
AVSA.to_csv('Outputs/AVSA.csv', index=False)
LArSA.to_csv('Outputs/LArSA.csv', index=False)
LAcSA.to_csv('Outputs/LAcSA.csv', index=False)
QCSA.to_csv('Outputs/QCSA.csv', index=False)
TASA.to_csv('Outputs/TASA.csv', index=False)
MCSA.to_csv('Outputs/MCSA.csv', index=False)
# %%Strata Analysis output part 2
StrataAnalysis = [AVSA, LArSA, LAcSA, QCSA, TASA, MCSA]
sheet_names = ['AVSA', 'LArSA', 'LAcSA', 'QCSA', 'TASA', 'MCSA']

# Create a Pandas Excel writer object
with pd.ExcelWriter('Outputs/StrataAnalysis.xlsx', engine='xlsxwriter') as writer:
    for StrataAnalysis, sheet in zip(StrataAnalysis, sheet_names):
        StrataAnalysis.to_excel(writer, sheet_name=sheet, index=False)
# %% Geospatial Analysis
MapData = result.copy()
MapData['predicted_log_Assessment_Val'] = regresult.predict(MapData)
MapData['predicted_Assessment_Val'] = np.exp(MapData['predicted_log_Assessment_Val'])
MapData['predicted_Market_Val'] = MapData['predicted_Assessment_Val'] + MapData['Total_MISC_Val']
MapData['Market_Residual'] = MapData['predicted_Market_Val'] - MapData['sl_price']
MapData['Assessment_Residual'] = MapData['predicted_Assessment_Val'] - MapData['Assessment_Val']
MapData['Market_Residual'] = pd.to_numeric(MapData['Market_Residual'], errors='coerce')
MapData['Assessment_Residual'] = pd.to_numeric(MapData['Assessment_Residual'], errors='coerce')
MapData['AbsV_Market_Residual'] = MapData['Market_Residual'].abs()
MapData['AbsV_Assessment_Residual'] = MapData['Assessment_Residual'].abs()
MapData['sale_ratio'] = MapData['predicted_Market_Val'] / MapData['sl_price']
MapData.to_csv('MapData.csv', index=False)
# %%
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
