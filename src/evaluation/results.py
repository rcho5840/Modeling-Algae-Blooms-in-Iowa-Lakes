import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


def results(direc, X_test, y_test):
    
    for filename in os.listdir(direc):
        print(filename + " model: \n")
        with open('../conf/models/' + filename, 'rb') as f:
            model = pickle.load(f)
    
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Score:{score}")
        print("\n")
