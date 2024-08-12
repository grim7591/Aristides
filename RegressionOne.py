import pandas as pd
from PreProcessingFunctions import preprocess_data
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

market_areas_path = 'Data/normalizedMAs.csv'
data_path = 'Data/dp20.csv'
preprocess_data(data_path=data_path, market_areas_path=market_areas_path)

def run_regression(data, formula, test_size=0.2, random_state=42):
    """
    Perform linear regression using statsmodels.

    Parameters:
    - data: DataFrame containing the data to be used for modeling.
    - formula: String, the regression formula specifying the model.
    - test_size: Float, the proportion of the data to include in the test split.
    - random_state: Integer, seed for the random number generator.

    Returns:
    - regresult: The fitted regression results.
    """
    # Split the data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Fit the model
    regresult = smf.ols(formula=formula, data=train_data).fit()
    
    # Print the summary of the regression results
    print(regresult.summary())
    
    return regresult

if __name__ == "__main__":
    market_areas_path = 'Data/normalizedMAs.csv'
    data_path = 'Data/dp20.csv'
    
    # Preprocess the data
    processed_data = preprocess_data(market_areas_path, data_path)
    
    # Define your regression formula
    regressionFormula_2 = "np.log(Assessment_Val) ~ np.log(living_area)+np.log(legal_acreage)+np.log(percent_good)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO+Springtree_B+HighSprings_A+MidtownEast_C+swNewberry_B+MidtownEast_A+swNewberry_A+MidtownEast_B+HighSprings_F+WaldoRural_C+Springtree_A+Tioga_B+Tioga_A+swNewberry_C+MidtownEast_D+HighSprings_E+MidtownEast_E+HighSprings_D+Springtree_C+WaldoRural_A+WaldoRural_B+HighSprings_C+MidtownEast_F+in_subdivision+A+B+D+E+F"
    
    # Run the regression
    regresult = run_regression(processed_data, regressionFormula_2)