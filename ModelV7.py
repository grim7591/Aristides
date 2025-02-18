import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from IAAOFunctions import PRD, COD, PRB, weightedMean, PRBCI


def prepare_data(
    main_csv: str = 'Data/dp74m.csv',
    mls_csv_map: dict = None,
    exclusions_csv_map: dict = None,
    outlier_csv: str = None,
    acreage_cutoff: float = 1.0
) -> pd.DataFrame:
    """
    Loads the main dataset from `main_csv`.
    Optionally filters by MLS data and year-based exclusions.
    Optionally removes outliers listed in `outlier_csv`.
    Performs the standard feature engineering steps.
    Returns the final cleaned dataframe.
    """

    # -----------------------------------------------------------------
    # 1. Load main data and rename columns
    # -----------------------------------------------------------------
    print(f"[prepare_data] Loading main data from {main_csv}")
    df = pd.read_csv(main_csv)
    df.rename(columns={'Name': 'Market_Cluster_ID'}, inplace=True)

    # Ensure prop_id and geo_id are strings
    df['prop_id'] = df['prop_id'].astype(str)
    df['geo_id'] = df['geo_id'].astype(str)

    # -----------------------------------------------------------------
    # 2. Load & build MLS sets (if provided)
    #    e.g., {2021: 'Data/MLSData/2021MLSData.csv', 2022: '...', ... }
    # -----------------------------------------------------------------
    sale_year_to_mls = {}
    if mls_csv_map is not None:
        for yr, csvpath in mls_csv_map.items():
            mls_df = pd.read_csv(csvpath)
            # Convert 'Tax ID' to string for membership checking
            sale_year_to_mls[yr] = set(mls_df['Tax ID'].astype(str))

    # Create sale_year = prop_val_yr - 1
    df['sale_year'] = df['prop_val_yr'] - 1

    # Define keep_row logic
    def keep_row(row):
        # If ratio_cd == 2, automatically keep
        if row['sl_county_ratio_cd'] == 2:
            return True
        # Otherwise, only keep if row's geo_id is in the right MLS set
        sy = row['sale_year']
        if sy in sale_year_to_mls:
            return row['geo_id'] in sale_year_to_mls[sy]
        # If no match, exclude
        return False

    # Filter DF if an MLS map was provided
    if mls_csv_map is not None:
        print("[prepare_data] Filtering rows based on MLS membership...")
        df = df[df.apply(keep_row, axis=1)]

    # -----------------------------------------------------------------
    # 3. Apply year-based exclusions (if provided)
    #    e.g., {2021: 'Data/XXI_exclusions_f2.csv', 2022: '...', ...}
    # -----------------------------------------------------------------
    sale_year_to_exclusion = {}
    if exclusions_csv_map is not None:
        for yr, csvpath in exclusions_csv_map.items():
            excl_df = pd.read_csv(csvpath)
            # Convert prop_id to string
            excl_set = set(excl_df['prop_id'].astype(str))
            sale_year_to_exclusion[yr] = excl_set

        def is_excluded(row):
            sy = row['sale_year']
            if sy in sale_year_to_exclusion:
                return row['prop_id'] in sale_year_to_exclusion[sy]
            return False

        print("[prepare_data] Filtering out year-based exclusions...")
        df = df[~df.apply(is_excluded, axis=1)]

    # -----------------------------------------------------------------
    # 4. Remove outliers if `outlier_csv` is provided
    # -----------------------------------------------------------------
    if outlier_csv is not None:
        print(f"[prepare_data] Removing outliers from {outlier_csv}...")
        outliers_df = pd.read_csv(outlier_csv)
        outlier_ids = set(outliers_df['prop_id'].astype(str))
        df = df[~df['prop_id'].isin(outlier_ids)]

    # -----------------------------------------------------------------
    # 5. Fix specific sale_price overrides (miscoded in PACS)
    # -----------------------------------------------------------------
    print("[prepare_data] Correcting specific sale prices (hard-coded fixes)...")
    overrides = {
        '84296': 90000,
        '79157': 300000,
        '93683': 199800,
        '93443': 132500
    }
    for pid, val in overrides.items():
        df.loc[df['prop_id'] == pid, 'sl_price'] = val

    # -----------------------------------------------------------------
    # 6. Compute Assessment_Val, ensure not negative
    # -----------------------------------------------------------------
    print("[prepare_data] Computing 'Assessment_Val' field...")
    df['Assessment_Val'] = 0.85 * (df['sl_price'] - (df['MISC_Val'] / 0.85))
    df['Assessment_Val'] = df['Assessment_Val'].apply(lambda x: x if x > 0 else np.nan)

    # -----------------------------------------------------------------
    # 7. Create landiness
    # -----------------------------------------------------------------
    print("[prepare_data] Creating 'landiness' feature...")
    avg_legal_acreage = (df['legal_acreage'] * 43560).mean()
    df['landiness'] = (df['legal_acreage'] * 43560) / avg_legal_acreage

    # -----------------------------------------------------------------
    # 8. Subdivision code => binary
    # -----------------------------------------------------------------
    df['in_subdivision'] = df['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
    df.drop(columns=['abs_subdv_cd'], inplace=True)

    # -----------------------------------------------------------------
    # 9. Cap effective_age at 30; compute percent_good
    # -----------------------------------------------------------------
    df['effective_age'] = df['effective_age'].apply(lambda x: 30 if x > 30 else x)
    df['percent_good'] = 1 - (df['effective_age'] / 100)

    # -----------------------------------------------------------------
    # 10. Hard-coded adjustments for impro_det_quality_cd
    # -----------------------------------------------------------------
    df.loc[df['prop_id'].isin(['96615']), 'imprv_det_quality_cd'] = 1
    df.loc[df['prop_id'].isin(['96411', '13894', '8894']), 'imprv_det_quality_cd'] = 2
    df.loc[df['prop_id'].isin(['91562', '73909']), 'imprv_det_quality_cd'] = 3
    df.loc[df['prop_id'].isin(['19165']), 'imprv_det_quality_cd'] = 4

    # Linearize quality codes
    quality_map = {
        1: 0.1331291,
        2: 0.5665645,
        3: 1.0,
        4: 1.1624432,
        5: 1.4343298,
        6: 1.7062164
    }
    df['imprv_det_quality_cd'] = df['imprv_det_quality_cd'].replace(quality_map)

    # -----------------------------------------------------------------
    # 11. One-hot encoding
    # -----------------------------------------------------------------
    print("[prepare_data] Creating dummy variables for 'tax_area_description' & 'Market_Cluster_ID'...")
    df = df.join(pd.get_dummies(df['tax_area_description']))
    df = df.join(pd.get_dummies(df['Market_Cluster_ID']))

    # Rename columns that have spaces or special chars
    column_mapping = {
        'HIGH SPRINGS': 'HIGH_SPRINGS',
        "ST. JOHN'S": 'ST_JOHNS'
    }
    df.rename(columns=column_mapping, inplace=True)

    # -----------------------------------------------------------------
    # 12. Filter acreage
    # -----------------------------------------------------------------
    df = df[df['legal_acreage'] < acreage_cutoff]

    # Ensure columns are strings
    df.columns = df.columns.astype(str)

    return df
def train_and_evaluate(
    df: pd.DataFrame,
    formula: str,
    test_size: float = 0.2,
    random_seed: int = 44
):
    """
    Splits `df` into train/test, fits a statsmodels OLS using `formula`,
    prints out the summary, and calculates performance metrics.
    Returns the fitted model and a copy of the test data with predictions.
    """

    # 1) Train/Test Split
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_seed)

    # 2) Fit OLS model
    print("[train_and_evaluate] Fitting regression model...")
    model = smf.ols(formula=formula, data=train_data).fit()

    # 3) Print Summary
    print("[train_and_evaluate] Model Summary:")
    print(model.summary())

    # 4) Evaluate on test set
    test_data = test_data.copy()  # avoid SettingWithCopy issues
    test_data['predicted_log_Assessment_Val'] = model.predict(test_data)
    test_data['predicted_Assessment_Val'] = np.exp(test_data['predicted_log_Assessment_Val'])

    # Predicted market value
    predicted_values_market = test_data['predicted_Assessment_Val'] + test_data['MISC_Val']
    actual_values_market    = test_data['sl_price']

    # For the second MAE
    predicted_values_mae = test_data['predicted_Assessment_Val']
    actual_values_mae    = test_data['Assessment_Val']

    mae_1 = mean_absolute_error(predicted_values_market, actual_values_market)
    mae_2 = mean_absolute_error(predicted_values_mae, actual_values_mae)

    # IAAO metrics
    prd_val  = PRD(predicted_values_market, actual_values_market)
    cod_val  = COD(predicted_values_market, actual_values_market)
    prb_val  = PRB(predicted_values_market, actual_values_market)
    prbci_val= PRBCI(predicted_values_market, actual_values_market)
    wm_val   = weightedMean(predicted_values_market, actual_values_market)
    meanRatio= (predicted_values_market / actual_values_market).mean()
    medianRatio = (predicted_values_market / actual_values_market).median()

    # 5) Print results
    print("\n[train_and_evaluate] Performance Metrics:")
    print(f"   MAE (Market):         {mae_1}")
    print(f"   MAE (Assessment):     {mae_2}")
    print(f"   PRD:                  {prd_val}")
    print(f"   COD:                  {cod_val}")
    print(f"   PRB:                  {prb_val}")
    print(f"   PRBCI:                {prbci_val}")
    print(f"   WeightedMean:         {wm_val}")
    print(f"   meanRatio:            {meanRatio}")
    print(f"   medianRatio:          {medianRatio}\n")

    return model, test_data
def create_geospatial_output(df: pd.DataFrame, model, output_csv='MapData.csv'):
    """
    Applies `model` predictions to the entire `df` 
    and generates additional columns needed for geospatial mapping.
    Exports the result to CSV.
    """
    df = df.copy()  # to avoid mutating the original
    df['predicted_log_Assessment_Val'] = model.predict(df)
    df['predicted_Assessment_Val']     = np.exp(df['predicted_log_Assessment_Val'])
    df['predicted_Market_Val']        = df['predicted_Assessment_Val'] + df['MISC_Val']

    # Residuals
    df['Market_Residual']     = df['predicted_Market_Val'] - df['sl_price']
    df['Assessment_Residual'] = df['predicted_Assessment_Val'] - df['Assessment_Val']
    df['AbsV_Market_Residual']     = df['Market_Residual'].abs()
    df['AbsV_Assessment_Residual'] = df['Assessment_Residual'].abs()

    # Sale ratio
    df['sale_ratio'] = df['predicted_Market_Val'] / df['sl_price']

    print(f"[create_geospatial_output] Exporting geospatial data to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    return df
def main():
    # We’ll define the formula here:
    regressionFormula = (
        "np.log(Assessment_Val) ~ "
        "np.log(living_area) + np.log(landiness) + np.log(percent_good) + "
        "np.log(imprv_det_quality_cd) + np.log(total_porch_area + 1) + "
        "np.log(total_garage_area + 1) + number_of_baths + in_subdivision + "
        "C(Market_Cluster_ID)"
    )

    # Common dicts
    mls_csv_map = {
        2021: 'Data/MLSData/2021MLSData.csv',
        2022: 'Data/MLSData/2022MLSData.csv',
        2023: 'Data/MLSData/2023MLSData.csv',
        2024: 'Data/MLSData/2024MLSData.csv'
    }
    exclusions_csv_map = {
        2021: 'Data/XXI_exclusions_f3.csv',
        2022: 'Data/XXII_exclusions_f3.csv',
        2023: 'Data/XXIII_exclusions_f3.csv',
        2024: 'Data/XXIV_exclusions_f3.csv'
    }
    
    # -----------------------------------------------------------
    # Scenario 1: No outliers removed
    # -----------------------------------------------------------
    print("=== SCENARIO 1: No outlier removal ===")
    df_scenario1 = prepare_data(
        main_csv='Data/dp74m.csv',
        mls_csv_map=mls_csv_map,
        exclusions_csv_map=exclusions_csv_map,
        outlier_csv=None  # No outlier removal
    )
    model1, test_data1 = train_and_evaluate(df_scenario1, regressionFormula)
    create_geospatial_output(df_scenario1, model1, output_csv='MapData_scenario1.csv')

    # -----------------------------------------------------------
    # Scenario 2: Remove the 3IQR outliers
    # -----------------------------------------------------------
    print("=== SCENARIO 2: Remove 3IQR outliers ===")
    df_scenario2 = prepare_data(
        main_csv='Data/dp74m.csv',
        mls_csv_map=mls_csv_map,
        exclusions_csv_map=exclusions_csv_map,
        outlier_csv='3IQR.csv'
    )
    model2, test_data2 = train_and_evaluate(df_scenario2, regressionFormula)
    create_geospatial_output(df_scenario2, model2, output_csv='MapData_scenario2.csv')

    # ...You could define more scenarios if needed...
    # Scenario 3: Larger or smaller acreage cutoff, different outlier file, etc.

if __name__ == "__main__":
    main()
