import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IAAOFunctions import PRD, COD, PRB, weightedMean, averageDeviation, PRBCI
def prepare_data(
    data_path: str,
    mls_data_path: str = None,
    outlier_path: str = None,
    filter_mls: bool = False,
    filter_outliers: bool = False,
    acreage_cutoff: float = 1
) -> pd.DataFrame:
    """
    Loads data from CSV, optionally filters by MLS or outliers,
    and performs all feature engineering steps.
    """

    # 1. Load main data
    df = pd.read_csv(data_path)
    
    # 2. (Optional) Load MLS data if we need to filter
    if filter_mls and mls_data_path is not None:
        mls_df = pd.read_csv(mls_data_path)
        df = df[df['geo_id'].isin(mls_df['Tax ID'])]
    
    # 3. (Optional) Load outlier data if we need to filter
    if filter_outliers and outlier_path is not None:
        outliers = pd.read_csv(outlier_path)
        df = df[~df['prop_id'].astype(str).isin(outliers['prop_id'].astype(str))]
    
    # 4. Apply transformations
    df.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)
    
    # One-hot encoding for tax area
    df = df.join(pd.get_dummies(df['tax_area_description']))
    df = df.join(pd.get_dummies(df['Market_Cluster_ID']))
    
    # 5. Filter out large acreage
    df.drop(df[df['legal_acreage'] >= acreage_cutoff].index, inplace=True)
    
    # 6. Filter on Join_Count
    df.drop(df[df['Join_Count'] != 1].index, inplace=True)
    
    # 7. Engineer "landiness"
    avg_legal_acreage = (df['legal_acreage'] * 43560).mean()
    df['landiness'] = (df['legal_acreage'] * 43560) / avg_legal_acreage
    
    # 8. In_subdivision (binary)
    df['in_subdivision'] = df['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
    df.drop(columns=['abs_subdv_cd'], inplace=True)
    
    # 9. Cap effective_age at 30
    df['effective_age'] = df['effective_age'].apply(lambda x: 30 if x > 30 else x)
    
    # 10. Calculate percent good
    df['percent_good'] = 1 - (df['effective_age']/100)
    
    # 11. Linearize quality codes
    quality_map = {
        1: 0.1331291,
        2: 0.5665645,
        3: 1.0,
        4: 1.1624432,
        5: 1.4343298,
        6: 1.7062164
    }
    df['imprv_det_quality_cd'] = df['imprv_det_quality_cd'].replace(quality_map)

    # Make sure prop_id is string
    df['prop_id'] = df['prop_id'].astype(str)
    
    return df
def evaluate_model(df: pd.DataFrame, regresult) -> None:
    """
    Takes in a prepared dataframe and a trained regression model,
    applies predictions and prints out metrics.
    """
    # Predict log-transformed assessment
    df['predicted_log_Assessment_Val'] = regresult.predict(df)
    
    # Convert to original scale
    df['predicted_Assessment_Val'] = np.exp(df['predicted_log_Assessment_Val'])
    
    # Predicted total market value
    predicted_values_market = df['predicted_Assessment_Val'] + df['MISC_Val']
    actual_values_market    = df['sl_price']
    
    # For assessment MAE
    #   (Your code that does .85*(sl_price - MISC_val/.85))
    df['Assessment_Val']  = .85 * (df['sl_price'] - (df['MISC_Val'] / .85))
    
    # Evaluate
    mae_market = mean_absolute_error(predicted_values_market, actual_values_market)
    print(f"MAE (Market): {mae_market}")
    
    mae_assessment = mean_absolute_error(df['predicted_Assessment_Val'], df['Assessment_Val'])
    print(f"MAE (Assessment): {mae_assessment}")
    
    # IAAO metrics
    prd_result = PRD(predicted_values_market, actual_values_market)
    cod_result = COD(predicted_values_market, actual_values_market)
    prb_result = PRB(predicted_values_market, actual_values_market)
    prbci_result = PRBCI(predicted_values_market, actual_values_market)
    
    # Weighted mean, meanRatio, medianRatio
    wm_val = weightedMean(predicted_values_market, actual_values_market)
    mean_ratio = (predicted_values_market / actual_values_market).mean()
    median_ratio = (predicted_values_market / actual_values_market).median()
    
    print(f"PRD: {prd_result}")
    print(f"COD: {cod_result}")
    print(f"PRB: {prb_result}")
    print(f"PRBCI: {prbci_result}")
    print(f"weightedMean: {wm_val}")
    print(f"meanRatio: {mean_ratio}")
    print(f"medianRatio: {median_ratio}")
    
    # If you want to return them for further usage:
    # return {
    #     'MAE_market': mae_market,
    #     'MAE_assessment': mae_assessment,
    #     'PRD': prd_result,
    #     'COD': cod_result,
    #     'PRB': prb_result,
    #     'PRBCI': prbci_result,
    #     'weightedMean': wm_val,
    #     'meanRatio': mean_ratio,
    #     'medianRatio': median_ratio
    # }
def iqr_outliers(df: pd.DataFrame, ratio_col: str = 'sale_ratio', multiplier: float = 3) -> pd.DataFrame:
    """
    Returns the subset of df that is considered outliers based on `ratio_col`
    using the specified IQR multiplier.
    """
    q1 = df[ratio_col].quantile(0.25)
    q3 = df[ratio_col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = df[(df[ratio_col] < lower_bound) | (df[ratio_col] > upper_bound)]
    return outliers
def main(regresult):
    # Scenario A: Non-MLS filtered
    print("=== Non-MLS Filtered ===")
    df_non_mls = prepare_data(
        data_path='Data/oopsall24sales_2m.csv',
        mls_data_path='Data/MLSData/2024MLSData.csv',
        filter_mls=False,
        filter_outliers=False
    )
    evaluate_model(df_non_mls, regresult)
    
    # Scenario B: MLS filtered
    print("\n=== MLS Only ===")
    df_mls = prepare_data(
        data_path='Data/oopsall24sales_2m.csv',
        mls_data_path='Data/MLSData/2024MLSData.csv',
        filter_mls=True,       # Only difference: filter by MLS
        filter_outliers=False
    )
    evaluate_model(df_mls, regresult)
    
    # Scenario C: 3 * IQR filtered (Non-MLS)
    print("\n=== 3 * IQR Filtered ===")
    df_iqr = prepare_data(
        data_path='Data/oopsall24sales_2m.csv',
        outlier_path='3IQRXXIV_NoMLS.csv',  # The CSV with outliers
        filter_mls=False,
        filter_outliers=True
    )
    evaluate_model(df_iqr, regresult)
    
    # Scenario D: MLS + 3 * IQR
    print("\n=== MLS + 3 * IQR Filtered ===")
    df_mls_iqr = prepare_data(
        data_path='Data/oopsall24sales_2m.csv',
        mls_data_path='Data/MLSData/2024MLSData.csv',
        outlier_path='3IQRXXIV_MLS.csv',
        filter_mls=True,
        filter_outliers=True
    )
    evaluate_model(df_mls_iqr, regresult)
if __name__ == "__main__":
    # Assume you already have `regresult` (the fitted regression model) somewhere
    main(regresult)
