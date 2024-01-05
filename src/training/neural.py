from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import pickle

def neural(X_train, y_train):
    
    model = Sequential()
    model.add(Dense(25, input_dim = 6, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['R2Score'])
    model.summary()

    model.fit(X_train, y_train, validation_split = 0.2, epochs = 200)
    
    with open ('../conf/nnmodels/nn', 'wb') as f:
        pickle.dump(model, f)
    