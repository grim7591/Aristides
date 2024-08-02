# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
# %%
market_areas = pd.read_csv('MarketAreaPull.csv')
data_2 = pd.read_csv("dp17.csv")
fixed_dupes = pd.read_csv('fixed_dupes.csv')
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
ma_cleaned = pd.concat([ma_cleaned, fixed_dupes])
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
result = result.join(pd.get_dummies(result.tax_area_description)).drop(['tax_area_description'], axis=1)
# %%
result = result.join(pd.get_dummies(result.Market_Area)).drop(['Market_Area'], axis=1)
# %%
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# %%
result = result.drop(columns=['abs_subdv_cd'])
# %%
result.columns = result.columns.astype(str)
# %%
#----------------------------------
# Mass appraisal functions
#----------------------------------

def weightedMean(pred, sp):
    '''
    Returns the weighted mean ratio
    
        Parameters:
            pred (pandas.Series): Series of predicted values
            sp   (pandas.Series): Series of sale prices
            
        Returns:
            weighted mean (numpy.float64): Weighted mean ratio
            
    '''
    return pred.sum() / sp.sum()

def averageDeviation(pred, sp):
    '''
    Returns the average deviation
    
        Parameters:
            pred (pandas.Series): Series of predicted values
            sp   (pandas.Series): Series of sale prices
            
        Returns:
            average deviation (numpy.float64): Average difference between each value
            
    '''
    medianRatio = (pred / sp).median()
    return ((pred / sp) - medianRatio).abs().sum() / len(sp) 

def COD(pred, sp):
    '''
    Returns the coefficient of dispersion
    
        Parameters:
            pred (pandas.Series): Series of predicted values
            sp   (pandas.Series): Series of sale prices
            
        Returns:
            coefficient of dispersion (numpy.float64): Average deviation as a percentage
            
    '''
    medianRatio = (pred / sp).median()
    return (100.00 * averageDeviation(pred, sp)) / medianRatio

def PRD(pred, sp):
    '''
    Returns the price related differential
    
        Parameters:
            pred (pandas.Series): Series of predicted values
            sp   (pandas.Series): Series of sale prices
            
        Returns:
            price related differential (numpy.float64): Statistic for measuring assessment regressivity
            
    '''
    meanRatio = (pred / sp).mean()
    return meanRatio / weightedMean(pred, sp)

def PRB(pred, sp):
    '''
    Returns the price related bias
    
        Parameters:
            pred (pandas.Series): Series of predicted values
            sp   (pandas.Series): Series of sale prices
            
        Returns:
            price related bias results (dict): Dictionary containing the PRB statistic and it's significance
            
    '''
    RATIO = pred / sp
    medianRatio = (RATIO).median()
    VALUE = (0.50 * sp) + (0.50 * pred / pred.median())
    LN_VALUE = np.log(VALUE) / np.log(2)
    PCT_DIFF = (RATIO - medianRatio) / medianRatio
    modelData = sm.add_constant(LN_VALUE)
    model = sm.OLS(PCT_DIFF, modelData).fit()
    return {"PRB" : model.params[0], "Sig" : model.pvalues[0]}


# %%
#------------------------------------
# Helper functions for this notebook
#------------------------------------

def plotResults(data, error_column):
    '''
    Creates plots showing COD, PRD, PRB against percentage of corrupted data 
    
        Parameters:
            data          (pandas.DataFrame): DataFrame of model values
            error_column  (string): Name of column that contains the corrupted data
            
        Returns:
            None
            
    '''
    for stat in ["COD", "PRD", "PRB"]:
        p = sns.lmplot(x='Percent Corrupted', y=stat, data = data, lowess = True, line_kws={'color': 'red'})
        p.fig.set_figwidth(15)
        p.fig.set_figheight(2)
        p.ax.set_title("Simulated %s with increasing data corruption of %s" % (stat, error_column))
        p.ax.ticklabel_format(useOffset=False)


def model_corrupted_data(data, model_formula, error_column, percent_corrupted, error_mean, error_sd, column_min_value, column_max_value):
    '''
    Captures statistics for a regression model after randomly adding errors to a given coefficient
    
        Parameters:
            data              (pandas.DataFrame): The modeling data
            model_formula     (string):           Regression fromula
            error_column      (string):           The coefficient that with recieve the errors
            percent_corrupted (float):            The percentage of the data to corrupt
            error_mean        (float):            The center for the error distribution
            error_sd          (float):            Spread of the generated errros
            column_min_value  (float):            Minimum error generated value (example: Building SQFT should be > 0)
            column_max_value  (float):            Maximum error generated value
            
            
        Returns:
            rv (pandas.DataFrame):  A dataframe containing the Percent Corrupted, COD, PRD, PRB for the regression model
                                    It is possible that given enough errors a solution for the regression is unable to be 
                                    found.  If no solution found the return value is None
            
        
            
    '''
    percent_corrupted = np.clip(percent_corrupted, .01, 1.0)
    df=data.copy()
    hasError = np.random.binomial(1, percent_corrupted, size=df.shape[0])
    error = np.random.normal(error_mean, error_sd, df.shape[0])
    df[error_column] = np.where(hasError == 1, np.clip(df[error_column] + error, column_min_value, column_max_value), df[error_column]) 
    try:
        regression = smf.ols(formula=model_formula, data=df).fit()
        sp = np.exp(regression.model.endog)
        pred = np.exp(regression.fittedvalues)
        prb = PRB(pred, sp)
        rv = pd.DataFrame({
            "Percent Corrupted" : [percent_corrupted],
            "COD" : [COD(pred, sp)],
            "PRD" : [PRD(pred, sp)],
            "PRB" : [prb['PRB'] if prb['Sig'] <= .05 else None]
        })
        rv.name = error_column 
        return rv
    except:
        return None              
# %%
data = result
# %%
column_mapping = {
    '1': 'A',
    '2': 'B',
    '3': 'C',
    '4': 'D',
    '5': 'E',
    '6': 'F',
    'HIGH SPRINGS' : 'HIGH_SPRINGS',
    "ST. JOHN'S" : 'ST_JOHNS'    
}
#  %%
data.rename(columns=column_mapping, inplace=True)    
# %%
regressionFormula = "np.log(Aessessment_Val) ~ np.log(living_area)+np.log(legal_acreage)+np.log(actual_age)+np.log(condition_cd)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO+A+C+D+E+F+in_subdivision"
# %%
regresult = smf.ols(formula=regressionFormula, data=data).fit()
regresult.summary()
# %%
regressionFormula_2 = "np.log(Aessessment_Val) ~ np.log(living_area * condition_cd)+np.log(legal_acreage)+np.log(actual_age)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+WALDO+A+C+D+E+F+in_subdivision"
# %%
plt.figure(figsize=(15,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

# %%
regressionFormula_3 = "np.log(Aessessment_Val) ~ np.log(living_area * condition_cd)+np.log(legal_acreage)+np.log(actual_age)+A+B+C+D+E+F+in_subdivision"
# %%
data_copy = sm.add_constant()
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
# %%
regressionFormula_3 = "np.log(Aessessment_Val) ~ np.log(living_area * condition_cd)+np.log(legal_acreage)+np.log(actual_age)+in_subdivision"
# %%
data_numeric_reduced = data_numeric.drop(columns=['ST_JOHNS','SUWANNEE', 'B'])
data_copy = sm.add_constant(data_numeric_reduced)
# %%
value_counts = ma_cleaned['Market_Area'].value_counts()
print(value_counts)
# %%
