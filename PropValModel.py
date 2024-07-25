# %% [markdown]
# # ACPA Property Data Model in Process documentation
# 
# **Initial steps**
# 
# - Figure out how to make a satisfying markdown document
# to be able to keep track of progress (Done)
# - Get data from PACs for model training
#   - Options:
#       - Use current year sales to train
#           - Probably not the move but might not be a bad place to start.
#       - Use time adjusted sales to train
#           - Time adjust using historical data, might not be too hard.
#           - One year of sales probably isn't enough because it biases the model towards 2023-like market conditions
#       - Use non time adjusted sales and year becomes a factor?
# - Identify less than 20 features to test for
#   - Examples from Lee:
#       - base area 
#       - age 
#       - neighborhood/market area 
#       - quality 
#       - land square ft
# - Do everything else
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("datapull1.csv") 

data

data.info() 
# %%

data.dropna(inplace=True)

data

data.info()
# %%
from sklearn.model_selection import train_test_split

X = data.drop(['sl_price'], axis = 1)
y = data['sl_price']
# %%
X
# %%
y
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
train_data = X_train.join(y_train)
# %%
train_data
# %%
train_data.hist(figsize =(15,8))
# %%
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# %%
train_data['legal_acreage'] = np.log(train_data['legal_acreage'] + 1)
train_data['sl_price'] = np.log(train_data['sl_price'] + 1)
train_data['living_area'] = np.log(train_data['living_area'] + 1)
# %%
train_data.hist(figsize=(15,8))
# %%
train_data.township_code.value_counts()
# %%
train_data.range_code.value_counts()
# %% [markdown]
# Need a location factor. Edit the MA spreadsheets to make Cluster ID unique then join into data, Real process starts here
# %%
market_area_1 = pd.read_csv('MA1_Export_20240719.csv')
market_area_2 = pd.read_csv('MA2_Export_20240719.csv')
market_area_3 = pd.read_csv('MA3_Export_20240719.csv')
data_2 = pd.read_csv("datapull2.csv")
# %%
market_area_1 = market_area_1[['prop_id', 'Cluster_ID']]
market_area_2 = market_area_2[['prop_id', 'Cluster_ID']]
market_area_3 = market_area_3[['prop_id', 'Cluster_ID']]
# %%
market_areas = pd.concat([market_area_1, market_area_2, market_area_3])
result = pd.merge(data_2, market_areas, how='left', on='prop_id')
# %%
print(result.columns)
print(result.dtypes)
# %%
result
# %% [markdown]
# Redo the data transfromation but maybe let's do it before we split out the test data because Im tired
# of doing things twice and this doc is going to get messy and need to be pruned. 
# %%
result.dropna(inplace=True)
# %%
result
# %%
plt.figure(figsize=(15,8))
sns.heatmap(result.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# %%
result = result.drop(['prop_id'], axis=1)
# %%
plt.figure(figsize=(15,8))
sns.heatmap(result.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# %%
result.hist(figsize =(15,8))
# %%
result=result.drop(['land_total_acres'], axis=1)
# %
# %%
result
# %%
result['legal_acreage'] = np.log(result['legal_acreage'] + 1)
result['sl_price'] = np.log(result['sl_price'] + 1)
result['living_area'] = np.log(result['living_area'] + 1)
result['yr_blt'] = np.log(result['yr_blt'] + 1)
# %%
result.hist(figsize =(15,8))
# %%
result.city_id.value_counts()
# %%
result.land_type_cd.value_counts()
# %%
result.Cluster_ID.value_counts()
# %%
plt.figure(figsize=(15,8))
sns.heatmap(result.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# %%
result = result.join(pd.get_dummies(result.city_id)).drop(['city_id'], axis=1)
# %%
result = result.join(pd.get_dummies(result.land_type_cd)).drop(['land_type_cd'], axis=1)
# %%
result.join(pd.get_dummies(result.Cluster_ID)).drop(['Cluster_ID'], axis=1)
# %%
plt.figure(figsize=(15,8))
sns.heatmap(result.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# %% [markdown]
# Weak correlation for individual city ID's but since 0 had some correaltion, presence in A city 
# seems more important than which city a prop is in. Running out of time today so Im going to run
# with this data for now but city_id and land_type_cd might be kind of useless.
# %%
from sklearn.model_selection import train_test_split

X = result.drop(['sl_price'], axis=1)
y = result['sl_price']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
train_data = X_train.join(y_train)
# %%
