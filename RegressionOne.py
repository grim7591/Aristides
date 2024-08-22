import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD,PRB, weightedMean, averageDeviation 
import numpy as np

def run_regression(data, formula):
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
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Fit the model
    regresult = smf.ols(formula, data=train_data).fit()
    
    # Print the summary of the regression results
    print(regresult.summary())
    
    return regresult, test_data



def evaluate_model(regresult, test_data):
    predictions = test_data.copy()
    predictions['predicted_log_Assessment_Val'] = regresult.predict(predictions)
    predictions['predicted_Assessment_Val'] = np.exp(predictions['predicted_log_Assessment_Val'])
    actual_values = predictions['Assessment_Val']
    predicted_values = predictions['predicted_Assessment_Val']
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    PRD_table = PRD(actual_values, predicted_values)
    COD_table = COD(actual_values, predicted_values)
    PRB_table = PRB(actual_values, predicted_values)
    wm = weightedMean(actual_values, predicted_values)
    ad = averageDeviation(actual_values, predicted_values)


    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"PRD: {PRD_table}")
    print(f"COD: {COD_table}")
    print(f"PRB: {PRB_table}")
    print(f"weightedMean: {wm}")
    print(f"averageDevitation: {ad}")
    
    return  {
        "MAE": mae,
        "MSE": mse,
        "R-squared": r2,
        "PRD": PRD_table,
        "COD": COD_table,
        "PRB": PRB_table,
        "Weighted Mean": wm,
        "Average Deviation": ad
    }