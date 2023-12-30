import seaborn as sns 
def distributions(data):
    a = sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu")
    b = data.hist(figsize = (15,8))
    return a, b
