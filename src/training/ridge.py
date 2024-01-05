from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import pickle

def ridge(X_train, y_train):
    ridge_reg = linear_model.Ridge()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    ridge_grid_search = GridSearchCV(ridge_reg, param_grid, cv = 10, scoring = 'neg_mean_squared_error')
    ridge_grid_search.fit(X_train, y_train)
    best_ridge = ridge_grid_search.best_estimator_
    
    with open ('../conf/models/best_ridge', 'wb') as f:
        pickle.dump(best_ridge, f)
    