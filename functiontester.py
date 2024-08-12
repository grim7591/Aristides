from PreProcessingFunctions import preprocess_data, load_data, preprocess_market_areas
import pandas as pd

preprocess_data('Data/dp20.csv','Data/normalizedMAs.csv')

# Preprocess the data
sale_data, market_areas =load_data('Data/dp20.csv','Data/normalizedMAs.csv')

print(market_areas)

processedMAs = preprocess_market_areas(market_areas)