from sklearn.ensemble import VotingRegressor
import os
import pickle

def ensemble(list, X_train, y_train):
    
    for filename in list:
        with open('../conf/models/' + filename, 'rb') as f:
            model = pickle.load(f)
            
        if filename == 'best_random_forest':
            best_random_forest = model
            
        elif(filename == 'best_lasso'):
            best_lasso = model
        else: 
            best_ridge = model
        
    er = VotingRegressor([('forr', best_random_forest), ('lasr', best_lasso), ('ridg', best_ridge)])
    er.fit(X_train, y_train)
    
    with open ('../conf/models/ensemble', 'wb') as f:
        pickle.dump(er, f)
    

    




# param_grid = {'weights' : [(1,1,1), (1,2,1), (1,1,2), (2,1,1), (1, 2, 2), (2,2,1)]}
# ensemble_grid_search = GridSearchCV(er, param_grid)
# ensemble_grid_search.fit(X_train_s, y_train)

# best_en = ensemble_grid_search.best_estimator_


