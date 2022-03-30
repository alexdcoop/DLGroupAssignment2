from sklearn.model_selection import GridSearchCV

def param_tune(estimator):
    param_grid = [{'batch_size':[], 'num_hid_layers':[], 'num_hid_neurons':[], 'hid_activation':['sigmoid', 'tanh','relu','elu','leaky relu','elu'], 'optimizer':['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']}]

    grid_search = GridSearchCV(estimator, param_grid,cv = 5, )

    return grid_search.best_params_



# Grid search
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# There are still some things below we will have to change. I think we need to add the metric=accuracy to the param_tune or model. Which optimizes the accuracy of the model

# Setting the seed (we may delete this later)
seed = 50
np.random.seed(seed)
# load dataset
pricing = pd.read_csv('pricing.csv')

# split into input (X) and output (Y) variables
X1 = pricing["sku"]
X2 = pricing["price"]
X3 = pricing["order"]
X4 = pricing["duration"]
X5 = pricing["category"]
X = np.array(np.column_stack((X1,X2,X3,X4,X5)))    
Y = pricing['quantity']
# create model
model = KerasClassifier(build_fn=param_tune, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
num_hid_layers = [5, 10, 15, 20, 25, 30]
num_hid_neurons = [20, 40, 80, 120, 160, 200]
epochs = [10, 25, 50, 100, 125, 150]
param_grid = dict(batch_size=batch_size,num_hid_layers = num_hid_layers, num_hid_neurons=num_hid_neurons, epochs=epochs )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
