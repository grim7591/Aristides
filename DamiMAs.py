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
result = pd.read_csv('Data/dp70m3.csv')
DamiMAs = pd.read_csv('Data/Regrsn_2024.csv')
result = result.merge(DamiMAs[['prop_id', 'SUBMKT']], on='prop_id', how='left')
result.rename(columns={'SUBMKT': 'Market_Cluster_ID'}, inplace=True)
result = result[~result['Market_Cluster_ID'].isin(['Highsprings_B10', 'swNewberry_C1'])]
result = result.dropna()
#result = pd.read_csv('Data/dp2124c.csv')
#result.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
#flood_zones = pd.read_csv('Data/Prop_FloodPlains.csv')
#flood_zones['prop_id'] = flood_zones['prop_id'].astype(str)
result['prop_id'] = result['prop_id'].astype(str)
#result = result.merge(flood_zones[['prop_id', 'FLD_ZONE']], on='prop_id', how='left')

# %% [markdown]
# ### Overwriting the sale price of some properties whose sales were miscoded in PACs

# %%
result.loc[result['prop_id'] == '84296', 'sl_price'] = 90000
result.loc[result['prop_id'] == '79157', 'sl_price'] = 300000
result.loc[result['prop_id'] == '93683', 'sl_price'] = 199800
result.loc[result['prop_id'] == '93443', 'sl_price'] = 132500

# %% [markdown]
# ### Creating "Assessment_Val"
# Assessment Value = 0.85 * (sale price - (MISC_Val/0.85)). This is the value the model will try to predict. Per statute we should aim to assess at 85% of purchase price to account for closing costs. MISC value is removed from the value used for training and testing because the model only accounts for the lot and the base improvement, it has no way to meaningfully interpret and predict MISC value. I believe the MISC values that we have in PACs come from cost manuals. 

# %%
# Factor engineer "Assessment Val"
print("Factor engineering Assessment Val...")
# Calculate the 'Assessment_Val' based on the sale price and miscellaneous value
result['Assessment_Val'] = .85 * (result['sl_price'] - (result['MISC_Val'] / .85))
# Add a validation step to ensure 'Assessment_Val' is not negative
result['Assessment_Val'] = result['Assessment_Val'].apply(lambda x: x if x > 0 else np.nan)

# %% [markdown]
# ### Creating "landiness" 
# landiness = legal_acreage / avg_legal_acreage. I also converted everything to square feet but I can't remember why.
# 

# %%
# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (result['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
result['landiness'] = (result['legal_acreage'] * 43560) / avg_legal_acreage
# %% [markdown]
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# %%
# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
result = result.drop(columns=['abs_subdv_cd'])

# Convert 'prop_id' to string for consistency across dataframes
result['prop_id'] = result['prop_id'].astype(str)

# %% [markdown]
# ### Effective age overwrites
# In 2024 we updated the effective year built of all properties to 1994 at minimum. When reviewing outliers I applied that same logic to these properties which were evaluated on pre-2024 factors. It was determined by valuation that a 30 year limit on effective age makes sense and because that was a change in our process and not necessarily a market shift, I think it makes sense to mitigate the impact of that change on the model by applying it retroactively to previous sale years. 

# %%
result['effective_age'] = result['effective_age'].apply(lambda x: 30 if x > 30 else x)

# %% [markdown]
# ### Calculating "percent good" from effective age
# Percent good = 1 - (effective_age/100)

# %%
# Factor Engineer Percent Good based on effective age
print("Calculating percent good based on effective age...")
# Calculate 'percent_good' as a factor of effective age
result['percent_good'] = 1 - (result['effective_age']/ 100)

# %% [markdown]
# ### Quality code overwrites
# When reviewing outliers I found some properties that needed to have their quality codes overwritten. As far as I understand it, because these are prior year factors, we are not able to edit them in PACs so I edit them here.

# %%
result.loc[result['prop_id'].isin(['96615']), 'imprv_det_quality_cd'] = 1

result.loc[result['prop_id'].isin(['96411', '13894', '8894']), 'imprv_det_quality_cd'] = 2

result.loc[result['prop_id'].isin(['91562', '73909']), 'imprv_det_quality_cd'] = 3

result.loc[result['prop_id'].isin(['19165']), 'imprv_det_quality_cd'] = 4

# %% [markdown]
# ### Linearize the quality codes
# Still working with Michael on this. For now I'm using what is already in PACs but here are the two linearizations we've been testing.
# 
#     1: 0.75,
#     2: 0.90,
#     3: 1.00,
#     4: 1.15,
#     5: 1.40,
#     6: 1.70
# 
#     1: 0.1331291,
#     2: 0.5665645,
#     3: 1.0,
#     4: 1.1624432,
#     5: 1.4343298,
#     6: 1.7062164
# %% [markdown]
# ### Adding handcrafted Market Cluster ID's
# Using the results of the model as guide we futher subdivided the properties in the initial market clusters based on comparable valuation, geography, tax areas, well defined and/or unique neighborhoods, etc. These are still in progress at the moment.
# %% [markdown]
# ## Market Clusters map and summary statistics
# These clusters are still a work in progress

# %%
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display
from IPython.display import HTML


# Use the provided MapData as IntMapData
IntMapData = result.copy()

# Calculate the mean latitude and longitude
center_lat = IntMapData['CENTROID_y'].mean()
center_lon = IntMapData['CENTROID_X'].mean()

# Create the map centered on the mean location
map_clusters = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='OpenStreetMap',
    width='80%',
    height='600px'
)

# Generate 40 unique colors using the updated Matplotlib colormaps API
cmap = plt.colormaps["tab20b"]  # Tab20b colormap for distinct colors
color_list = [mcolors.rgb2hex(cmap(i / 20)) for i in range(20)] * 2  # Repeat to ensure 40 colors

# Assign unique colors to each Market_Cluster_ID
unique_clusters = IntMapData['Market_Cluster_ID'].unique()
cluster_colors = {
    cluster_id: color for cluster_id, color in zip(
        unique_clusters, color_list[:len(unique_clusters)]
    )
}

# Automatically calculate map bounds
min_lat, max_lat = IntMapData['CENTROID_y'].min(), IntMapData['CENTROID_y'].max()
min_lon, max_lon = IntMapData['CENTROID_X'].min(), IntMapData['CENTROID_X'].max()

# Fit map to bounds dynamically based on displayed points
def update_bounds(points):
    latitudes = points['CENTROID_y']
    longitudes = points['CENTROID_X']
    return [[latitudes.min(), longitudes.min()], [latitudes.max(), longitudes.max()]]

bounds = update_bounds(IntMapData)
map_clusters.fit_bounds(bounds)

# Create feature groups for each market cluster
for cluster_id in unique_clusters:
    cluster_group = folium.FeatureGroup(name=f"Cluster {cluster_id}", show=False)  # Default to off

    # Add properties to the respective cluster
    cluster_data = IntMapData[IntMapData['Market_Cluster_ID'] == cluster_id]
    for _, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=[row['CENTROID_y'], row['CENTROID_X']],
            radius=3,  # Smaller size for better visualization
            color=cluster_colors.get(cluster_id, 'gray'),
            fill=True,
            fill_color=cluster_colors.get(cluster_id, 'gray'),
            fill_opacity=0.8,
            popup=f"<strong>Market Area:</strong> {row['Market_Cluster_ID']}"
        ).add_to(cluster_group)

    # Add the cluster group to the map
    map_clusters.add_child(cluster_group)

# Add a layer control to toggle groups
folium.LayerControl(collapsed=False).add_to(map_clusters)

# Inject JavaScript for dynamic autozoom functionality
map_clusters.get_root().html.add_child(folium.Element("""
<script>
document.addEventListener("DOMContentLoaded", function() {
    var mapElement = document.querySelector('div[id^="map_"]'); // Find the map element
    if (mapElement) {
        var mapId = mapElement.id; // Get the map's unique ID
        var mapInstance = window[mapId]; // Access the map object

        function updateMapBounds() {
            var bounds = new L.LatLngBounds();
            mapInstance.eachLayer(function (layer) {
                if (layer instanceof L.LayerGroup && mapInstance.hasLayer(layer)) {
                    layer.eachLayer(function (subLayer) {
                        if (subLayer.getLatLng) {
                            bounds.extend(subLayer.getLatLng());
                        }
                    });
                }
            });
            if (!bounds.isValid()) {
                return;
            }
            mapInstance.fitBounds(bounds);
        }

        // Attach the updateMapBounds function to layer events
        mapInstance.on('overlayadd', updateMapBounds);
        mapInstance.on('overlayremove', updateMapBounds);
    }
});
</script>
"""))

# Display the map inline
display(HTML(map_clusters._repr_html_()))


# %%
'''
# Undo the linearization of quality codes
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    0.1331291: 1,
    0.5665645: 2,
    1.0: 3,
    1.1624432: 4,
    1.4343298: 5,
    1.7062164: 6
})
'''
# Group by Market_Cluster_ID to calculate metrics
summary_stats = result.groupby('Market_Cluster_ID').agg(
    count=('Market_Cluster_ID', 'size'),
    median_actual_age=('actual_age', 'median'),
    median_living_area=('living_area', 'median'),
    median_sl_price=('sl_price', 'median'),
    mode_condition_quality_cd=('imprv_det_quality_cd', lambda x: x.mode().iloc[0] if not x.mode().empty else None)
).reset_index()

# Display the summary stats table
from IPython.display import display
print("Summary Statistics Table:")
display(summary_stats.style.hide(axis='index'))
# %% [markdown]
# ### Creating dummy variables for non-numeric data
# Since several of the factors are not numbers we need to make binary dummy variables so the regression can recognize their impact. Dummy variables put different values for a given factor into categories and each property is either 1 or 0 (true or false) for each category. Some of the market clusters also have to have their names slightly tweaked so that python will behave.

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
# %%
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

# %% [markdown]
# ### Large acerage exclusion
# The maximum legal acreage for the model is set to `legalAcreageMax`, which is currently 10 acres.

# %%
# Define the variable
legalAcreageMax = 1  # in acres

result = result[result['legal_acreage'] < legalAcreageMax]

# %% [markdown]
# ### Ensuring all column names are strings
# Data type mismatches cause annoying errors

# %%
# Ensure that all column names are strings
result.columns = result.columns.astype(str)
# %% [markdown]
# ## Regression
# ### Formula
# I decided to go with a log based regression formula because most of the numeric factors we use are not normally distributed. Here it is in markdown for the sake of readability:
# 
# regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + Springtree_B + HighSprings_A + MidtownEast_C + swNewberry_B + MidtownEast_A + swNewberry_A + MidtownEast_B + HighSprings_F + Springtree_A + Tioga_B + Tioga_A + MidtownEast_D + WaldoRural_A + Alachua_Main + High_Springs_Main + HaileLike + HighSprings_B + Real_Tioga + Duck_Pond + Newmans_Lake + EastMidtownEastA + HighSpringsAGNV + Hawthorne + HighSprings_B + Golfview + Lugano + Archer + WildsPlantation+Buck_Bay+in_subdivision+has_lake+WaldoRural_C+HighSprings_E+HSBUI+number_of_baths+EastGNV+Ironwood+SummerCreek+has_canal+TC_Forest+CarolEstates+Westchesterish+QuailCreekish"

# %%
#regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + number_of_baths + in_subdivision + Alachua_Main + Archer + Buck_Bay + Carol_Estates + Duck_Pond + EastGNV + EastMidtownEastA + Golfview + Haile_Like + Hawthorne + Hickory_Forest + High_Springs_Main + HighSpringsA + HighSpringsAGNV + HighSpringsAGNVSouth + HighSpringsB+Florida_Park + Ironwood + Jonesville + Kanapaha + Lincoln_Estates + Lugano + MidtownEast_A + MidtownEast_B + MidtownEast_C + Montery  + Newnans_Lake + QuailCreek + Rural_South + San_Felasco + Sorrento + Split_Rock + Springtree + SummerCreek + Sweetwater + swNewberry_A + swNewberry_B + TC_Forest + Tioga + Tioga_A + Tioga_B + Turkey_Creek + Valwood + Waldo + WaldoRural + Westchester + WildsPlantation+Highland"

regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + number_of_baths + in_subdivision + C(Market_Cluster_ID)"
# %% [markdown]
# ### Train/Test Split
# The data is split into training and testing sets. The training data is used to inform the model, the test data is used to check the performance of the trained model. The split takes out a random 20% of the properties to use for testing but for my purposes I've been using the same random seed so that variation in the results is from changes I make to the model and not from just getting a different seed. I believe the plan in the future will be to run the model on multiple seeds.

# %%
# Split data into training and test sets
print("Splitting data into training and test sets...")
test_size_var = 0.2
train_data, test_data = train_test_split(result, test_size=test_size_var, random_state=42)

# %% [markdown]
# ### Regression run
# This is where the regression is actually run and the statistical summary generated.

# %%
# Fit the regression model
print("Fitting the regression model...")
regresult = smf.ols(formula=regressionFormula, data=train_data).fit()
# Display regression summary
print("Regression model summary:")
print(regresult.summary())

# %% [markdown]
# ### Evaluating model performance with appraisal metrics
# The model generated by the training data and regression is used to predict the assessment value of the test data properties and then compared to the actual sale price of those properties in order to evaluate performance. The goal is to achieve a sale ratio of 0.85 in line with local laws and appraisal standards.
# 
# Metrics evaluated:
# 
# - Mean absolute error: Measures the average absolute difference between the predicted total (Assessment Value + MISC Value) and the actual sale price.
# 
# - Mean absolute error 2: Measures the average absolute difference between the predicted assessment values and the actual assessment values (calculated as Sale Price - MISC Value). The target value for both measures of MAE is as close to zero as possible. 
# 
# - Price Related Differential: Measures assessment equity; calculated as the mean assessment ratio divided by the weighted mean assessment ratio. Per IAAO standards a PRD value between 0.98 and 1.03 is considered acceptable. 
# 
# - Coefficient of Dispersion: Measures the average absolute percentage deviation of the ratios from the median ratio. Per IAAO standards a COD value between 5.0 and 10.0 is considered acceptable.  
# 
# - Price related bias: The PRB provides a percentage by which
# model-derived value estimates rise or fall as values double.Per IAAO standards, the PRB coefficient should fall between –0.05 and 0.05.
# 
# - Weighted Mean: Averages the ratios of assessment values to sale prices, weighting by the sale price. Used to evaluate overall equity in assessments. The aim is to be as close to 0.85 as possible.
# 
# - Mean sales ratio: The unweighted average of assessment-to-sale price ratios across all properties, providing a simple measure of assessment equity. The aim is to be as close to 0.85 as possible.
# 
# - Median sales ratio: The middle value of the assessment-to-sale price ratios when sorted in order. Often preferred for its resistance to outliers and skewed distributions. The aim is to be as close to 0.85 as possible.
# 

# %%
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
# 2024 only predictions
# PREDICTIONS
XXIVSales = pd.read_csv('Data/oopsall24sales.csv')
MLS_SalesXXIV = pd.read_csv('Data/2024MLSData.csv')
XXIVOutliers = pd.read_csv('3IQRXXIV_2.csv')
XXIVSales = XXIVSales[XXIVSales['geo_id'].isin(MLS_SalesXXIV['Tax ID'])]
XXIVSales = XXIVSales[~XXIVSales['prop_id'].isin(XXIVOutliers['prop_id'])]
#XXIVSales.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales.merge(DamiMAs[['prop_id', 'SUBMKT']], on='prop_id', how='left')
XXIVSales.rename(columns={'SUBMKT': 'Market_Cluster_ID'}, inplace=True)
XXIVSales = XXIVSales[~XXIVSales['Market_Cluster_ID'].isin(['Highsprings_B10', 'swNewberry_C1'])]
XXIVSales = XXIVSales.dropna()
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.tax_area_description))
XXIVSales = XXIVSales.join(pd.get_dummies(XXIVSales.Market_Cluster_ID))
XXIVSales.drop(XXIVSales[XXIVSales['legal_acreage'] >= 1].index, inplace=True)
XXIVSales.drop(XXIVSales[XXIVSales['Market_Cluster_ID'] == 'Rural_North'].index, inplace=True)
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
outliers_dfXXIV.to_csv('3IQRXXIV_2.csv')
# %%
