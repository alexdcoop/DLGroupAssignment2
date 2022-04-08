#This is the final model file
from asyncio.unix_events import BaseChildWatcher
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from ann_visualizer.visualize import ann_viz

#1. Prepare data
# Read in data
DATA = pd.read_csv('pricing.csv')

# Split into X and Y
Y = DATA.pop('quantity')
X = np.array(np.column_stack((DATA['sku'],DATA['price'],DATA['order'],DATA['duration'],DATA['category'])))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


# def createNeuralNet(learning_rate, num_hid_layers, num_hid_neurons, hid_activation, optimizer):
#     inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
#     hidden = tf.keras.layers.Dense(units=num_hid_neurons, activation=hid_activation)(inputs)
    
#     for layer in range(num_hid_layers):
#         hidden = tf.keras.layers.Dense(units=num_hid_neurons, activation=hid_activation)(hidden)

#     output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden)
    
#     #Create model 
#     model = tf.keras.Model(inputs = inputs, outputs = output)
    
#     #make opimizer
#     if optimizer == 'plain SGD':
#         optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate)
#     elif optimizer == 'momentum':
#         optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = .9) #Hardcoded optimizer
#     elif optimizer == 'nesterov':
#         optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = .9, nesterov=True)
#     elif optimizer == 'adagrad':
#         optimizerCode = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
#     elif optimizer == 'rmsprop':
#         optimizerCode = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=.9,momentum=0.0,epsilon=1e-07)
#     elif optimizer == 'adam':
#         optimizerCode = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9,beta_2=.999,epsilon=1e-07)
#     elif optimizer == 'learning rate scheduling':
#         initial_learning_rate = learning_rate; decay_steps = 10000; decay_rate = .95
#         learning_sched = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps, decay_rate)
#         optimizerCode = tf.keras.optimizers.SGD(learning_rate=learning_sched)
#     else:
#         print('Error making optimizer')
#         return -1

#     #Compile model
#     model.compile(loss = 'mse', optimizer = optimizerCode)

#     #return model
#     return model

#2. Create a grid of parameter values    
#Make Grid of all possible parameters
#param_grid = [{'learning_rate':[.001,.01,.1,.2,.3], 'num_hid_layers':[5,4,3,2,1], 'num_hid_neurons':[75, 50, 25, 10, 5], 'hid_activation':['sigmoid', 'tanh','relu','elu'], 'optimizer':['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']}]

learning_rate = [.001,.01,.1,.2,.3]
num_hid_layers = [5,4,3,2,1]
num_hid_neurons=[75, 50, 25, 10, 5]
hid_activation=['sigmoid', 'tanh','relu','elu']
optimizer=['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']
initializer=['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform']

param_grid= dict(learning_rate = learning_rate, num_hid_layers=num_hid_layers, num_hid_neurons=num_hid_neurons, hid_activation=hid_activation,optimizer=optimizer, initializer=initializer)

#3. Create a function that takes values of the grid, and returns a compiled model architecture 
def createModel(learning_rate, num_hid_layers, num_hid_neurons, hid_activation, optimizer, initializer):
    #Create model
    model = keras.Sequential()
    
    #Add input layer
    model.add(keras.Input(shape=(X.shape[1],)))
    
    #Add dense layers
    for layer in range(num_hid_layers):
        model.add(layers.Dense(units=num_hid_neurons, activation=hid_activation, kernel_initializer=initializer))
        
    #output layer
    model.add(layers.Dense(units=1, activation = "softmax",kernel_initializer=initializer))
    
    #make opimizer
    if optimizer == 'plain SGD':
        optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    elif optimizer == 'momentum':
        optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = .9) #Hardcoded optimizer
    elif optimizer == 'nesterov':
        optimizerCode = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = .9, nesterov=True)
    elif optimizer == 'adagrad':
        optimizerCode = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
    elif optimizer == 'rmsprop':
        optimizerCode = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=.9,momentum=0.0,epsilon=1e-07)
    elif optimizer == 'adam':
        optimizerCode = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9,beta_2=.999,epsilon=1e-07)
    elif optimizer == 'learning rate scheduling':
        initial_learning_rate = learning_rate; decay_steps = 10000; decay_rate = .95
        learning_sched = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps, decay_rate)
        optimizerCode = tf.keras.optimizers.SGD(learning_rate=learning_sched)
    else:
        print('Error making optimizer')
        return -1
    
    #compile model
    model.compile(loss = 'mae', optimizer=optimizerCode)
    
    return model

#Batch sizes
{'batch_size':[100000, 50000, 10000, 5000, 1000]}

model = KerasRegressor(build_fn=createModel, epochs=10, batch_size=100000, verbose=0)

#Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid)
gridResult = grid.fit(X,Y)


# summarize results
print("Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))
means = gridResult.cv_results_['mean_test_score']
stds = gridResult.cv_results_['std_test_score']
params = gridResult.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#Random search
rnd_search = RandomizedSearchCV(model, param_grid, n_iter =20, cv=3)
rnd_search.fit(X,Y)

# summarize results
print("Best: %f using %s" % (rnd_search.best_score_, rnd_search.best_params_))
means = rnd_search.cv_results_['mean_test_score']
stds = rnd_search.cv_results_['std_test_score']
params = rnd_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
model = createModel(.2,3,5,'tanh','rmsprop','glorot_normal')

model.fit(X_train, Y_train,epochs=10,batch_size=10000)
model.summary()

y = model.predict(X_test)

#5. Plot results

ann_viz(model, title="Our Neural Network", filename="dlproj2.gv", view = True) #should produce a visual of the neural net