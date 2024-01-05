from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle

def svr(X_train, y_train):
    
    svr = svm.SVR()

    param_grid = {
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'C' : [0.1, 1, 10]
    }

    svr_grid_search = GridSearchCV(svr, param_grid, cv = 10, scoring = 'neg_mean_squared_error')
    svr_grid_search.fit(X_train, y_train)

    best_svr = svr_grid_search.best_estimator_

    with open ('../conf/models/best_svr', 'wb') as f:
        pickle.dump(best_svr, f)
