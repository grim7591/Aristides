# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
market_area_1 = pd.read_csv('MA1_Export_20240719.csv')
market_area_2 = pd.read_csv('MA2_Export_20240719.csv')
market_area_3 = pd.read_csv('MA3_Export_20240719.csv')
data_2 = pd.read_csv("dp14.csv")
# %%
market_area_1 = market_area_1[['prop_id', 'Cluster_ID']]
market_area_2 = market_area_2[['prop_id', 'Cluster_ID']]
market_area_3 = market_area_3[['prop_id', 'Cluster_ID']]
# %%
market_areas = pd.concat([market_area_1, market_area_2, market_area_3])
result = pd.merge(data_2, market_areas, how='left', on='prop_id')
# %%
result.dropna(inplace=True)
# %%
result = result.drop(['prop_id'], axis=1)
# %%
result
# %%
result['legal_acreage'] = np.log(result['legal_acreage'])
result['sl_price'] = np.log(result['sl_price'])
result['living_area'] = np.log(result['living_area'])
result['actual_age'] = np.log(result['actual_age'])
# %%
result = result.join(pd.get_dummies(result.tax_area_description)).drop(['tax_area_description'], axis=1)
# %%
result = result.join(pd.get_dummies(result.land_type_cd)).drop(['land_type_cd'], axis=1)
# %
result = result.join(pd.get_dummies(result.Cluster_ID)).drop(['Cluster_ID'], axis=1)
# %%
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# %%
result = result.drop(columns=['abs_subdv_cd'])
# %%
result.columns = result.columns.astype(str)
# %%
from sklearn.model_selection import train_test_split
X = result.drop(['sl_price'], axis=1)
y = result['sl_price']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
from sklearn.linear_model import LinearRegression
# %%
reg = LinearRegression()
# %%
reg.fit(X_train, y_train)
# %%
reg.score(X_test, y_test)
# %%
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg.coef_
})
print(coef_df)
# %%