import statsmodels.api as sm

class LinearModel:
    def __init__(self, data):
        self.data = data

    def fit(self, formula):
        model = sm.formula.ols(formula=formula, data=self.data)
        result = model.fit()
        return result
