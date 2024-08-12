# Import libraries
import pandas as pd

data_path = 'Data/dp20.csv'
market_area_path = 'Data/normalizedMAs.csv'

# Load the data
def load_data(data_path, market_area_path):
    sale_data = pd.read_csv(data_path)
    market_areas = pd.read_csv(market_area_path)
    return sale_data, market_areas

# Preprocess sale data: Engineer sale price into assessment value, ensure string formatting for merge
def preprocess_sale_data(sale_data):
    sale_data['Assessment_Val'] =.85 * (sale_data['sl_price'] - (sale_data['Total_MISC_Val']/.85))
    sale_data['prop_id'] = sale_data['prop_id'].astype(str)
    return sale_data

# Preprocess the Market Area data: remove null values and extraneous data and ensures string format for merge with sale data
def preprocess_market_areas(market_areas):
    market_areas = market_areas[['prop_id', 'MA', 'Cluster ID']]
    market_areas.dropna(inplace=True)
    market_areas = market_areas[market_areas['MA'] != '<Null>']
    market_areas = market_areas[market_areas['prop_id'] != '<Null>'] 
    market_areas['Market_Cluster_ID'] = market_areas['MA'].astype(str) + '_' + market_areas['Cluster ID'].astype(str)
    market_areas['prop_id'] = market_areas['prop_id'].astype(str)
    market_areas['Market_Cluster_ID'] = market_areas['Market_Cluster_ID'].astype(str)
    return market_areas

# Merge market areas with sale data and drop nulls from merged table
def merge_data(sale_data, market_areas):
    result = pd.merge(sale_data, market_areas, how='inner', on='prop_id')
    result.dropna(inplace=True)
    return result

# Encode categorical features
def encode_categorical_features(result):
    result = result.join(pd.get_dummies(result.imprv_det_quality_cd)).drop(['imprv_det_quality_cd'], axis=1)
    result = result.join(pd.get_dummies(result.tax_area_description)).drop(['tax_area_description'], axis=1)
    result = result.join(pd.get_dummies(result.Market_Cluster_ID)).drop(['Market_Cluster_ID'], axis=1)
    return result

# Rename coulmns so they play nice with python
def rename_columns(result):
    column_mapping = {
        'HIGH SPRINGS' : 'HIGH_SPRINGS',
        "ST. JOHN'S" : 'ST_JOHNS',
        '1.0' : 'A',
        '2.0' : 'B',
        '3.0' : 'C',
        '4.0' : 'D',
        '5.0' : 'E',
        '6.0' : 'F'
    }
    result.rename(columns=column_mapping, inplace=True)
    return result

# Engineer subdivision into a binary category and effective age into percent good. Drop extra columns.
def add_features(result):
    result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
    result['percent_good'] = 1 - (result['effective_age'] / 100)
    data = result.drop(columns=['abs_subdv_cd', 'MA', 'Cluster ID', 'sl_price', 'Total_MISC_Val', 'effective_age'])
    return data

# Final combined function
def preprocess_data(data_path, market_areas_path):
    sale_data, market_areas = load_data(data_path, market_areas_path)
    sale_data = preprocess_sale_data(sale_data)
    market_areas = preprocess_market_areas(market_areas)
    result = merge_data(sale_data, market_areas)
    result = encode_categorical_features(result)
    result = rename_columns(result)
    data = add_features(result)
    data.columns = data.columns.astype(str)
    return data