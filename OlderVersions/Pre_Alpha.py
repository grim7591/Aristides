# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
market_areas = pd.read_csv('MarketAreaPull.csv')
data_2 = pd.read_csv("dp17.csv")
# %%
market_areas = market_areas[['prop_id', 'Market_Area']]
# %%
market_areas.dropna(inplace=True)
# %%
market_areas = market_areas[market_areas['Market_Area'] != '<Null>']
market_areas = market_areas[market_areas['prop_id'] != '<Null>']
# %%
ma_unique = market_areas.drop_duplicates()
# %%
duplicates = ma_unique[ma_unique.duplicated(subset='prop_id', keep=False)]
# %%
ma_cleaned = ma_unique[~ma_unique['prop_id'].isin(duplicates['prop_id'])].copy()
# %%
data_2['prop_id'] = data_2['prop_id'].astype(str)
ma_cleaned['prop_id'] = ma_cleaned['prop_id'].astype(str)
# %%
result = pd.merge(data_2, ma_cleaned, how='inner', on='prop_id')
# %%
result.dropna(inplace=True)
# %%
result = result.drop(['prop_id'], axis=1)
# %%
result
# %%
result['legal_acreage'] = np.log(result['legal_acreage'])
result['Aessessment_Val'] = np.log(result['Aessessment_Val'])
result['living_area'] = np.log(result['living_area'])
result['actual_age'] = np.log(result['actual_age'])
# %%
result = result.join(pd.get_dummies(result.tax_area_description)).drop(['tax_area_description'], axis=1)
# %%
#result = result.join(pd.get_dummies(result.land_type_cd)).drop(['land_type_cd'], axis=1)
# %%
result = result.join(pd.get_dummies(result.Market_Area)).drop(['Market_Area'], axis=1)
# %%
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# %%
result = result.drop(columns=['abs_subdv_cd'])
# %%
result.columns = result.columns.astype(str)
# %%
from sklearn.model_selection import train_test_split
X = result.drop(['Aessessment_Val'], axis=1)
y = result['Aessessment_Val']
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
