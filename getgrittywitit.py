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
market_areas = pd.read_csv('MarketAreaPull2.csv')
data_2 = pd.read_csv("dp18.csv")
# %%
market_areas = market_areas[['prop_id', 'Market Area', 'Cluster ID']]
# %%
market_areas.dropna(inplace=True)
# %%
market_areas = market_areas[market_areas['Market Area'] != '<Null>']
market_areas = market_areas[market_areas['prop_id'] != '<Null>']
# %%
market_areas['Market_Cluster_ID'] = market_areas['Market Area'].astype(str) + market_areas['Cluster ID'].astype(str)
# %%
data_2['prop_id'] = data_2['prop_id'].astype(str)
market_areas['prop_id'] = market_areas['prop_id'].astype(str)
market_areas['Market_Cluster_ID'] = market_areas['Market_Cluster_ID'].astype(str)
# %%
result = pd.merge(data_2, market_areas, how='inner', on='prop_id')
# %%
result.dropna(inplace=True)
# %%
result = result.drop(['prop_id'], axis=1)
# %%
result = result.join(pd.get_dummies(result.tax_area_description)).drop(['tax_area_description'], axis=1)
# %%
result = result.join(pd.get_dummies(result.Market_Cluster_ID)).drop(['Market_Cluster_ID'], axis=1)
# %%
result['in_subdivision'] = result['abs_subdv_cd'].apply(lambda x: True if x > 0 else False)
# %%
result = result.drop(columns=['abs_subdv_cd', 'Market Area', 'Cluster ID'])
# %%
result.columns = result.columns.astype(str)
# %%
from sklearn.model_selection import train_test_split
X = result.drop(['Aessessment_Val'], axis=1)
y = result['Aessessment_Val']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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
    'HIGH SPRINGS' : 'HIGH_SPRINGS',
    "ST. JOHN'S" : 'ST_JOHNS',
    '1A' : 'AA',
    '1B' : 'AB',
    '1C' : 'AC',
    
    '2A' : 'BA',
    '2B' : 'BB',
    '2C' : 'BC',

    '3A' : 'CA',
    '3B' : 'CB',
    '3C' : 'CC',

    '4A' : 'DA',
    '4B' : 'DB',
    '4C' : 'DC',

    '5A' : 'EA',
    '5B' : 'EB',
    '5C' : 'EC',

    '6A' : 'FA',
    '6B' : 'FB',
    '6C' : 'FC',  
    }
#  %%
data.rename(columns=column_mapping, inplace=True)    
# %%
regressionFormula = "np.log(Aessessment_Val) ~ np.log(living_area)+np.log(legal_acreage)+np.log(actual_age)+np.log(imprv_det_quality_cd)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+NEWBERRY+WALDO+AA+AB+AC+BA+BC+CA+CB+DA+DB+DC+EA+EB+EC+FA+FB+FC+in_subdivision"
# %%
regresult = smf.ols(formula=regressionFormula, data=data).fit()
regresult.summary()
# %%
## Everything after this is me just messing with other ways of doing this and evaluating factors and such
regressionFormula_2 = "np.log(Aessessment_Val) ~ np.log(living_area * imprv_det_quality_cd)+np.log(legal_acreage)+np.log(actual_age)+ALACHUA+ARCHER+GAINESVILLE+HAWTHORNE+HIGH_SPRINGS+WALDO+in_subdivision"
# %%
plt.figure(figsize=(15,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

# %%
regressionFormula_3 = "np.log(Aessessment_Val) ~ np.log(living_area * imprv_det_quality_cd)+np.log(legal_acreage)+np.log(actual_age)+A+B+C+D+E+F+in_subdivision"
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
# %%
regressionFormula_3 = "np.log(Aessessment_Val) ~ np.log(living_area * imprv_det_quality_cd)+np.log(legal_acreage)+np.log(actual_age)+in_subdivision"
# %%
data_numeric_reduced = data_numeric.drop(columns=['ST_JOHNS','SUWANNEE', 'B'])
data_copy = sm.add_constant(data_numeric_reduced)
# %%
value_counts = result['Market_Cluster_ID'].value_counts()
print(value_counts)
# %%
# %%
from sklearn.linear_model import LinearRegression
# %%
reg = LinearRegression()
# %%
reg.fit(X_train, y_train)
# %%
reg.score(X_test, y_test)
# %%