import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

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
    regresult = smf.ols(formula, data=train_data).fit()
    
    # Print the summary of the regression results
    print(regresult.summary())
    
    return regresult