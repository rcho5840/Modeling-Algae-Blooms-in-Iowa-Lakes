from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle

def lasso(X_train, y_train, X_test, y_test):
    lasso_reg = linear_model.Lasso()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    lasso_grid_search = GridSearchCV(lasso_reg, param_grid, cv = 10, scoring = 'neg_mean_squared_error')
    lasso_grid_search.fit(X_train, y_train)
    best_lasso = lasso_grid_search.best_estimator_
    
    with open ('../conf/best_lasso', 'wb') as f:
        pickle.dump(best_lasso, f)
    
    with open('../conf/best_lasso', 'rb') as f:
        model = pickle.load(f)
    
    
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Score:{score}")
    return score, model