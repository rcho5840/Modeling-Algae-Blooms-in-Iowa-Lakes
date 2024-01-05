from sklearn.linear_model import LinearRegression
import pickle
def baseline(X_train, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    with open ('../conf/models/baseline', 'wb') as f:
        pickle.dump(lin_reg, f)