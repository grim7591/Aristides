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
# %%
