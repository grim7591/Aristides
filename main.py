from data_processing import DataLoader, DataPreprocessor
from models import LinearModel

def main():
    # Paths to data files
    data_path = 'Data/dp20.csv'
    market_area_path = 'Data/normalizedMAs.csv'

    # Load and preprocess data
    loader = DataLoader(data_path, market_area_path)
    sale_data, market_areas = loader.load_data()
    
    preprocessor = DataPreprocessor(sale_data, market_areas)
    processed_data = preprocessor.preprocess()
    
    # Train model
    formula = 'np.log(Assessment_Val) ~ np.log(living_area)'
    model = LinearModel(processed_data)
    result = model.fit(formula)
    
    print(result.summary())

if __name__ == "__main__":
    main()