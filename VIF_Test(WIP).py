# %%
data_copy = sm.add_constant(data)
#data_copy = data_copy.select_dtypes(include=[np.number])
# %%
vif_data = pd.DataFrame()
vif_data["feature"] = data_copy.columns
vif_data["VIF"] = [vif(data_copy.values, i)
                   for i in range(data_copy.shape[1])]
# %%
print(vif_data)
# %%
data_numeric = data.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)