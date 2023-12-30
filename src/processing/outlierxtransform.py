from scipy import stats
import seaborn as sns
def outlier(data):
    print('old data:', len(data))
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.boxplot(data)
    for col in data.columns:
        upper_limit = data[col].mean() + 3*data[col].std()
        lower_limit = data[col].mean() - 3*data[col].std()
        data.drop(data[(data[col] > upper_limit) | (data[col] < lower_limit)].index, inplace = True)
        print('new data after outlier check in {0}: {1}'.format(col, len(data)))
        
    
    print('final new data', len(data))
    sns.boxplot(data)
    return data



def transform(data, list):
    data.hist(figsize = (15,8))
    for col in list:
        data[col], lambdavalue = stats.boxcox(data[col])
        print('Selected value for lambda is ', lambdavalue)
        data.hist(figsize = (15,8))
        return data