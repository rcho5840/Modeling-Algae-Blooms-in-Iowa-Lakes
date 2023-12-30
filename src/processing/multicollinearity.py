import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
def multicollinearity(df, target):
    X = df.drop(target, axis = 1)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return print(vif_data), df.corr()