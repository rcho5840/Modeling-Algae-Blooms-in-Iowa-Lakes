import seaborn as sns 
import pandas as pd
def distributions(data):
    a = sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu")
    b = data.hist(figsize = (15,8))
    return a, b
