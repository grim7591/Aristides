# %% [markdown]
# # Project Aristides
# ### A linear regression model for predicting the value of single family homes for tax purposes
# 
# ## Data Preprocessing
# 
# ### Data Pull
# 
# The model is fed three years of qualified 0100 property sales from 2021 to 2023, with property factors taken from the year following each sale.
# 
# Properties were excluded under several conditions:  
# - If there was a living area change greater than 50 square feet between the sale year and the subsequent year.  
# - If they were involved in splits or combines.  
# - If obsolescence was recorded in PACs, due to concerns over data quality and consistency.  
# - If the actual or effective year was greater than the sale date (new constructions).  
# - If they were assessed with a flat value source in PACs, raising concerns about the functionality of those improvements.  
# - If they had more than one improvement, as these were considered outside the scope of the model.
# 
# After the initial model run, properties with a sale ratio beyond three times the interquartile range (3 * IQR) were excluded as outliers, per IAAO standards. Properties with sale ratios beyond 1.5 * IQR were reviewed, then either excluded or corrected if necessary.
# 
# After the IQR evaluation, 208 properties were excluded. Additionally, 53 properties were marked for overrides, with 27 still pending review by subject matter experts.   
# 
# ### Property Factors
# The factors pulled from PACs to use for training and prediction are as follows:
# - legal_acreage
# - living_area
# - imprv_det_quality_cd
# - tax_area_description
# - abs_subdv_cd
# - sl_price
# - effective_year_built
# - imprv_type_cd
# - base_area
# - actual_year_built
# - prop_val_yr
# - total_porch_area (engineered in SQL)
# - total_garage_area (engineered in SQL)
# - effective_age (engineered in SQL)
# - has_canal (engineered in SQL)
# - has_lake (engineered in SQL)
# - number_of_baths (engineered in SQL)
# - MISC_Val
# 
# ### Market Areas
# Sean is most familiar with where these market areas came from. I believe we did some multivariate clustering in ArcGIS on the sale data. That got us market areas which were further subdivided into submarkets. This is the location component of the model. 

# %% [markdown]
# ### Library and data import
# Importing python libraries, all of the data, market areas, removing properties with null values, and making sure everything is in a format the model can work with.

# %%
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
import matplotlib.pyplot as plt
from IPython.display import Markdown

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
import matplotlib.pyplot as plt

# Load the data
print("Loading data from CSV files...")

# Load data from multiple CSV files
market_areas = pd.read_csv('Data/normalizedMAs.csv')
sale_data = pd.read_csv("Data/dp53.csv")

Haile = pd.read_csv("Data/Haile.csv")
High_Springs_Main = pd.read_csv("Data/High_Springs_Main.csv")
Turkey_Creek = pd.read_csv("Data/Turkey_Creek.csv")
Alachua_Main = pd.read_csv("Data/Alachua_Main.csv")
Gainesvilleish_Region = pd.read_csv("Data/Gainesvilleish_Region.csv")
Real_Tioga = pd.read_csv("Data/Real_Tioga.csv")
Duck_Pond = pd.read_csv("Data/DuckPond.csv")
Newmans_Lake = pd.read_csv("Data/Newmans_Lake.csv")
EastMidtownEastA = pd.read_csv("Data/EastMidtownEastA.csv")
HighSpringsAGNV = pd.read_csv("Data/HighSpringsAGNV.csv")
Golfview = pd.read_csv("Data/Golfview.csv")
Lugano = pd.read_csv("Data/Lugano.csv")
Archer = pd.read_csv("Data/Archer.csv")
WildsPlantation = pd.read_csv("Data/WildsPlantation.csv")
Greystone = pd.read_csv("Data/Greystone.csv")
Eagle_Point = pd.read_csv("Data/Eagle_Point.csv")
Near_Haile = pd.read_csv("Data/Near_Haile.csv")
Buck_Bay = pd.read_csv("Data/Buck_Bay.csv")
Ironwood = pd.read_csv("Data/Ironwood.csv")
Serenola = pd.read_csv("Data/Serenola.csv")
BluesCreek = pd.read_csv("Data/BluesCreek.csv")
Edgemoore = pd.read_csv("Data/Edgemoore.csv")
SummerCreek = pd.read_csv("Data/SummerCreek.csv")
EastGNV = pd.read_csv("Data/EastGNV.csv")
TC_Forest = pd.read_csv("Data/TC_Forest.csv")
CarolEstates = pd.read_csv("Data/CarolEstates.csv")
Westchesterish = pd.read_csv("Data/Westchesterish.csv")
QuailCreekish = pd.read_csv("Data/QuailCreekish.csv")
Gainesvilleish_Region_2 = pd.read_csv("Data/Gainesville_Region_2.csv")
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

# %% [markdown]
# ## Factor Engineering
# The next step is to modify some of the factors we took out of PACs to make them more useful for model training. 
# 
#  ### Creating Market_Cluster_ID
#  Functionally there is only one location component and it's "Market Cluster ID". These clusters were cut out from the submarkets established in previous steps.

# %%
# Factor engineer "Market Cluster ID"
print("Creating Market Cluster ID...")
# Create a new column 'Market_Cluster_ID' by combining 'MA' and 'Cluster ID'
market_areas['Market_Cluster_ID'] = market_areas['MA'].astype(str) + '_' + market_areas['Cluster ID'].astype(str)

# %% [markdown]
# ### Overwriting the sale price of some properties whose sales were miscoded in PACs

# %%
sale_data.loc[sale_data['prop_id'] == '84296', 'sl_price'] = 90000
sale_data.loc[sale_data['prop_id'] == '79157', 'sl_price'] = 300000
sale_data.loc[sale_data['prop_id'] == '93683', 'sl_price'] = 199800
sale_data.loc[sale_data['prop_id'] == '93443', 'sl_price'] = 132500

# %% [markdown]
# ### Creating "Assessment_Val"
# Assessment Value = 0.85 * (sale price - (MISC_Val/0.85)). This is the value the model will try to predict. Per statute we should aim to assess at 85% of purchase price to account for closing costs. MISC value is removed from the value used for training and testing because the model only accounts for the lot and the base improvement, it has no way to meaningfully interpret and predict MISC value. I believe the MISC values that we have in PACs come from cost manuals. 

# %%
# Factor engineer "Assessment Val"
print("Factor engineering Assessment Val...")
# Calculate the 'Assessment_Val' based on the sale price and miscellaneous value
sale_data['Assessment_Val'] = .85 * (sale_data['sl_price'] - (sale_data['MISC_Val'] / .85))
# Add a validation step to ensure 'Assessment_Val' is not negative
sale_data['Assessment_Val'] = sale_data['Assessment_Val'].apply(lambda x: x if x > 0 else np.nan)

# %% [markdown]
# ### Creating "landiness" 
# landiness = legal_acreage / avg_legal_acreage. I also converted everything to square feet but I can't remember why.
# 

# %%
# Factor engineer "landiness"
print("Calculating landiness...")
# Calculate the average legal acreage in square feet
avg_legal_acreage = (sale_data['legal_acreage'] * 43560).mean()
# Create 'landiness' as a ratio of property acreage to average acreage
sale_data['landiness'] = (sale_data['legal_acreage'] * 43560) / avg_legal_acreage

# %% [markdown]
# ### Merging the pulled sales data with the market area spreadsheet

# %%
# Merge the market area and sale data
print("Merging market area and sale data...")
# Merge sale_data and market_areas on 'prop_id'
result = pd.merge(sale_data, market_areas, how='inner', on='prop_id')
# Drop rows with missing values after merging
result.dropna(inplace=True)

# %% [markdown]
# ### Creating in_subdivision
# Binary variable for if a property is in a subdivision or not.

# %%
# Make subdivision code binary variable
print("Creating binary variables for subdivision status...")
# Create a binary variable 'in_subdivision' to indicate if property is in a subdivision
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# Drop unnecessary columns
result = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID'])

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

result.loc[result['prop_id'].isin(['96411']), 'imprv_det_quality_cd'] = 2

result.loc[result['prop_id'].isin(['13894']), 'imprv_det_quality_cd'] = 2

result.loc[result['prop_id'].isin(['8894']), 'imprv_det_quality_cd'] = 2

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

# %%
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

# %% [markdown]
# ### Adding handcrafted Market Cluster ID's
# Using the results of the model as guide we futher subdivided the properties in the initial market clusters based on comparable valuation, geography, tax areas, well defined and/or unique neighborhoods, etc. These are still in progress at the moment.

# %%
# New Market Area subdivisions
print("Updating Market Cluster IDs for new subdivisions...")

# Ensure 'prop_id' is a string for all subdivision dataframes
Haile['prop_id'] = Haile['prop_id'].astype(str)
High_Springs_Main['prop_id'] = High_Springs_Main['prop_id'].astype(str)
Turkey_Creek['prop_id'] = Turkey_Creek['prop_id'].astype(str)
Alachua_Main['prop_id'] = Alachua_Main['prop_id'].astype(str)
Gainesvilleish_Region['prop_id'] = Gainesvilleish_Region['prop_id'].astype(str)
Real_Tioga['prop_id'] = Real_Tioga['prop_id'].astype(str)
Duck_Pond['prop_id'] = Duck_Pond['prop_id'].astype(str)
Newmans_Lake['prop_id'] = Newmans_Lake['prop_id'].astype(str)
EastMidtownEastA['prop_id'] = EastMidtownEastA['prop_id'].astype(str)
HighSpringsAGNV['prop_id'] = HighSpringsAGNV['prop_id'].astype(str)
Golfview['prop_id'] = Golfview['prop_id'].astype(str)
Lugano['prop_id'] = Lugano['prop_id'].astype(str)
Archer['prop_id'] = Archer['prop_id'].astype(str)
WildsPlantation['prop_id'] = WildsPlantation['prop_id'].astype(str)
Greystone['prop_id'] = Greystone['prop_id'].astype(str)
Eagle_Point['prop_id'] = Eagle_Point['prop_id'].astype(str)
Near_Haile['prop_id'] = Near_Haile['prop_id'].astype(str)
Buck_Bay['prop_id'] = Buck_Bay['prop_id'].astype(str)
EastGNV['prop_id'] = EastGNV['prop_id'].astype(str)
SummerCreek['prop_id'] = SummerCreek['prop_id'].astype(str)
Ironwood['prop_id'] = Ironwood['prop_id'].astype(str)
TC_Forest['prop_id'] = TC_Forest['prop_id'].astype(str)
CarolEstates['prop_id'] = CarolEstates['prop_id'].astype(str)
Westchesterish['prop_id'] = Westchesterish['prop_id'].astype(str)
QuailCreekish['prop_id'] = QuailCreekish['prop_id'].astype(str)
Gainesvilleish_Region_2['prop_id'] = Gainesvilleish_Region_2['prop_id'].astype(str)

# Assign new Market Cluster IDs based on subdivision membership and tax area description
result.loc[result['prop_id'].isin(Haile['prop_id']), 'Market_Cluster_ID'] = 'HaileLike'
result.loc[result['tax_area_description'] == 'LACROSSE', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['tax_area_description'] == 'HAWTHORNE', 'Market_Cluster_ID'] = 'Hawthorne'
result.loc[result['Market_Cluster_ID'] == 'HighSprings_D', 'Market_Cluster_ID'] = 'High_Springs_Main'
result.loc[result['Market_Cluster_ID'] == 'MidtownEast_E', 'Market_Cluster_ID'] = 'MidtownEast_C'
result.loc[result['Market_Cluster_ID'] == 'MidtownEast_F', 'Market_Cluster_ID'] = 'MidtownEast_B'
result.loc[result['Market_Cluster_ID'] == 'HighSprings_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['Market_Cluster_ID'] == 'Springtree_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['Market_Cluster_ID'] == 'swNewberry_C', 'Market_Cluster_ID'] = 'HSBUI'
result.loc[result['prop_id'].isin(High_Springs_Main['prop_id']), 'Market_Cluster_ID'] = 'High_Springs_Main'
result.loc[result['prop_id'].isin(Turkey_Creek['prop_id']), 'Market_Cluster_ID'] = 'Turkey_Creek'
result.loc[result['prop_id'].isin(Alachua_Main['prop_id']), 'Market_Cluster_ID'] = 'Alachua_Main'
result.loc[result['prop_id'].isin(Gainesvilleish_Region['prop_id']), 'Market_Cluster_ID'] = 'Gainesvilleish_Region'
result.loc[result['prop_id'].isin(Real_Tioga['prop_id']), 'Market_Cluster_ID'] = 'Real_Tioga'
result.loc[result['prop_id'].isin(Duck_Pond['prop_id']), 'Market_Cluster_ID'] = 'Duck_Pond'
result.loc[result['prop_id'].isin(Newmans_Lake['prop_id']), 'Market_Cluster_ID'] = 'Newmans_Lake'
result.loc[result['prop_id'].isin(EastMidtownEastA['prop_id']), 'Market_Cluster_ID'] = 'EastMidtownEastA'
result.loc[result['prop_id'].isin(HighSpringsAGNV['prop_id']), 'Market_Cluster_ID'] = 'HighSpringsAGNV'
result.loc[result['prop_id'].isin(Golfview['prop_id']), 'Market_Cluster_ID'] = 'Golfview'
result.loc[result['prop_id'].isin(Lugano['prop_id']), 'Market_Cluster_ID'] = 'Lugano'
result.loc[result['prop_id'].isin(Archer['prop_id']), 'Market_Cluster_ID'] = 'Archer'
result.loc[result['prop_id'].isin(WildsPlantation['prop_id']), 'Market_Cluster_ID'] = 'WildsPlantation'
result.loc[result['prop_id'].isin(Greystone['prop_id']), 'Market_Cluster_ID'] = 'HaileLike'
result.loc[result['prop_id'].isin(Near_Haile['prop_id']), 'Market_Cluster_ID'] = 'HaileLike'
result.loc[result['prop_id'].isin(Buck_Bay['prop_id']), 'Market_Cluster_ID'] = 'Buck_Bay'
result.loc[result['prop_id'].isin(EastGNV['prop_id']), 'Market_Cluster_ID'] = 'EastGNV'
result.loc[result['prop_id'].isin(SummerCreek['prop_id']), 'Market_Cluster_ID'] = 'SummerCreek'
result.loc[result['prop_id'].isin(Ironwood['prop_id']), 'Market_Cluster_ID'] = 'Ironwood'
result.loc[result['prop_id'].isin(TC_Forest['prop_id']), 'Market_Cluster_ID'] = 'TC_Forest'
result.loc[result['prop_id'].isin(CarolEstates['prop_id']), 'Market_Cluster_ID'] = 'CarolEstates'
result.loc[result['prop_id'].isin(Westchesterish['prop_id']), 'Market_Cluster_ID'] = 'Westchesterish'
result.loc[result['prop_id'].isin(QuailCreekish['prop_id']), 'Market_Cluster_ID'] = 'QuailCreekish'
result.loc[result['prop_id'].isin(Gainesvilleish_Region_2['prop_id']), 'Market_Cluster_ID'] = 'Gainesvilleish_Region'

# Keep the first occurrence of each duplicate prop_id, only needed for PID 99411 which is duped for some reason
result = result.drop_duplicates(subset='prop_id', keep='first')
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
center_lat = IntMapData['CENTROID_Y'].mean()
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
min_lat, max_lat = IntMapData['CENTROID_Y'].min(), IntMapData['CENTROID_Y'].max()
min_lon, max_lon = IntMapData['CENTROID_X'].min(), IntMapData['CENTROID_X'].max()

# Fit map to bounds dynamically based on displayed points
def update_bounds(points):
    latitudes = points['CENTROID_Y']
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
            location=[row['CENTROID_Y'], row['CENTROID_X']],
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
# Undo the linearization of quality codes
result['imprv_det_quality_cd'] = result['imprv_det_quality_cd'].replace({
    0.75: 1,
    0.90: 2,
    1.00: 3,
    1.15: 4,
    1.40: 5,
    1.70: 6
})

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

# %%
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

# %% [markdown]
# ### Large acerage exclusion
# The maximum legal acreage for the model is set to `legalAcreageMax`, which is currently 10 acres.

# %%
# Define the variable
legalAcreageMax = 10  # in acres

result = result[result['legal_acreage'] <= legalAcreageMax]

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
regressionFormula = "np.log(Assessment_Val) ~ np.log(living_area) + np.log(landiness) + np.log(percent_good) + np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + np.log(total_garage_area + 1) + Springtree_B + HighSprings_A + MidtownEast_C + swNewberry_B + MidtownEast_A + swNewberry_A + MidtownEast_B + HighSprings_F + Springtree_A + Tioga_B + Tioga_A + MidtownEast_D + WaldoRural_A + Alachua_Main + High_Springs_Main + HaileLike + HighSprings_B + Real_Tioga + Duck_Pond + Newmans_Lake + EastMidtownEastA + HighSpringsAGNV + Hawthorne + HighSprings_B + Golfview + Lugano + Archer + WildsPlantation+Buck_Bay+in_subdivision+has_lake+WaldoRural_C+HighSprings_E+HSBUI+number_of_baths+EastGNV+Ironwood+SummerCreek+has_canal+TC_Forest+CarolEstates+Westchesterish+QuailCreekish"
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

# %% [markdown]
# # What's next?
# - The model struggles with extremely large homes (mansions) in rural areas.
# - Quality Code lineraization could probably be better
# - The market clusters still need some tweaking but I'm a little unsure where to go next with them.
# - Need subject matter experts to review some of the other 1.5 * IQR outliers
# - Need a way to assign market clusters to properties that aren't in the sale data for when we eventually deploy this. Working on a mix of random forest prediction model with some kind of spatial component and then some neighborhoods we could probably just cut by hand.
# - Whatever Michael tells me to do

# Here's the other junk I need that wasn't in the first report.
# %% Geospatial Analysis
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
# Group by Market_Cluster_ID and calculate summary statistics for sale ratio
summary_stats = MapData.groupby('Market_Cluster_ID')['sale_ratio'].describe()

# Add the median to the summary statistics
median_values = MapData.groupby('Market_Cluster_ID')['sale_ratio'].median()
summary_stats['median'] = median_values

# Display the updated summary statistics
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format  # Optional: Format floats for readability
print(summary_stats)

summary_stats.to_csv('summary_stats.csv')  # For CSV

import pandas as pd

# Ensure floats are displayed with 2 decimal places
pd.options.display.float_format = '{:.2f}'.format

# 1. Group by Quality Code
summary_stats_quality = MapData.groupby('imprv_det_quality_cd')['sale_ratio'].describe()
summary_stats_quality['median'] = MapData.groupby('imprv_det_quality_cd')['sale_ratio'].median()
summary_stats_quality.to_csv('summary_stats_quality.csv')
print(summary_stats_quality)
print("Summary stats by Quality Code saved as 'summary_stats_quality.csv'")

# 2. Group by Year Built
summary_stats_year = MapData.groupby('actual_year_built')['sale_ratio'].describe()
summary_stats_year['median'] = MapData.groupby('actual_year_built')['sale_ratio'].median()
summary_stats_year.to_csv('summary_stats_year.csv')
print(summary_stats_year)
print("Summary stats by Year Built saved as 'summary_stats_year.csv'")

# 3. Group by Tax Area
summary_stats_tax = MapData.groupby('tax_area_description')['sale_ratio'].describe()
summary_stats_tax['median'] = MapData.groupby('tax_area_description')['sale_ratio'].median()
summary_stats_tax.to_csv('summary_stats_tax.csv')
print(summary_stats_tax)
print("Summary stats by Tax Area saved as 'summary_stats_tax.csv'")
# %%
