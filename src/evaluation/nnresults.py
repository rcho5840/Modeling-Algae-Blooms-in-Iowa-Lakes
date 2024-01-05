import os
import pickle

def nnresults(direc, X_test, y_test):
    
    for filename in os.listdir(direc):
        print(filename + " model: \n")
        with open('../conf/nnmodels/' + filename, 'rb') as f:
            model = pickle.load(f)
        print(model.evaluate(X_test, y_test))
        print("\n")
