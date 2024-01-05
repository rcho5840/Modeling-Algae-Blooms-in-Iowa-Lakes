from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import pickle

def random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth' : [None, 4, 8],
        'min_samples_split': [2, 4, 6]
        
    }
    forest_grid_search = GridSearchCV(ensemble.RandomForestRegressor(), param_grid, cv = 10, scoring = 'neg_mean_squared_error')
    forest_grid_search.fit(X_train, y_train)
    best_random_forest = forest_grid_search.best_estimator_
    
    with open ('../conf/models/best_random_forest', 'wb') as f:
        pickle.dump(best_random_forest, f)
    