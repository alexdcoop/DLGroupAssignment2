#This is the final model file
#from asyncio.unix_events import BaseChildWatcher
from sklearn.metrics import log_loss
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
import matplotlib as plt
import itertools

#1. Prepare data
# Read in data
DATA = pd.read_csv('pricing.csv')
#DATA.head()

# Split into X and Y
Y = DATA.pop('quantity')
X = np.array(np.column_stack((DATA['sku'],DATA['price'],DATA['order'],DATA['duration'],DATA['category'])))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

glorot_sigm_init = keras.initializers.VarianceScaling(scale = 16, mode = 'fan_avg', distribution = 'uniform')
he_avg_init = keras.initializers.VarianceScaling(scale = 2, mode = 'fan_avg', distribution ='uniform')
#Define what goes into the grid
learning_rate = [.001,.01,.1,.2,.3, 0.02, 0.2]
num_hid_layers = [5,4,3,2,1, 7, 6]
num_hid_neurons=[75, 50, 25, 10, 5, 3, 2]
hid_activation=['sigmoid', 'tanh','relu','elu']
optimizer=['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']
initializer=['glorot_sigm_init', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'he_avg_init']
#param_grid= dict(learning_rate = learning_rate, num_hid_layers=num_hid_layers, num_hid_neurons=num_hid_neurons, hid_activation=hid_activation,optimizer=optimizer, initializer=initializer)

#make grid with all possible combinations
a = [learning_rate,num_hid_layers,num_hid_neurons,hid_activation,optimizer, initializer]
allCombinations = list(itertools.product(*a))
grid = pd.DataFrame(allCombinations)
colnames = ['learning_rate', 'num_hid_layers' , 'num_hid_neurons', 'hid_activation', 'optimizer', 'initializer']
grid.columns = colnames
grid['loss'] = ''
#grid.head()

#Define Model
def createModel(learning_rate, num_hid_layers, num_hid_neurons, hid_activation, optimizer, initializer):
    #Create model
    model = keras.Sequential()
    
    #Add input layer
    model.add(keras.Input(shape=(X.shape[1],)))
    
    #Add dense layers
    for layer in range(num_hid_layers):
        model.add(layers.Dense(units=num_hid_neurons, activation=hid_activation, kernel_initializer=initializer))
        
    #output layer
    model.add(layers.Dense(units=1, activation = "linear",kernel_initializer=initializer))
    
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

#Loops

batchSize = 10000
epochs = 10
for record in grid: #each record in grid represents a model version
    model = createModel(learning_rate = record[0], num_hid_layers = record[1], num_hid_neurons = record[2], hid_activation = record[3], optimizer = record[4], initializer = record[5]) #create model with inputs corresponding to the index location
    loss_array = [] #set up the list to store average losses in
    for _ in range(epochs): #for each of the 10 epochs
        for _ in range(len(record)): #this is the subsetted data that gets pulled into the lower part of the for loop to train the model 
            model.fit(epochs=1, batch_size=1) #could also do mini batch
        for _ in range(100):
            model.fit(x=X_train,y=y_train, batch_size=50, epochs=1,validation_data=(X_train,y_train)) 
            yhat_val = model.predict(x=X_val)
            loss_val = np.mean((y_val - yhat_val.flatten())**2)
            yhat_train = model.predict(x=X_train)
            loss_train = np.mean((y_train - yhat_train.flatten())**2)
        make a prediction on entire validation set and compute average log_loss
        loss_array.append(avg_loss) #append average loss to an array additional column in grid 
    store loss array in last column of grid grid['new column'] = loss_array (keep a counter)    


for record in grid: 
    model = createModel(learning_rate = record[0], num_hid_layers = record[1], num_hid_neurons = record[2], hid_activation = record[3], optimizer = record[4], initializer = record[5])
    loss_array = []
    for _ in range(epochs):
        for _ in range(len(X_train)): #this is the subsetted data that gets pulled into the lower part of the for loop to train the model 
            model.fit(X_train, Y_train, epochs=1, batch_size=batchSize) #could also do mini batch
        #make a prediction on entire validation set and compute average log_loss
        yhat = model.predict(x=X_test)
        avg_loss = np.mean(yhat)
        loss_array.append(avg_loss) #append average loss to an array additional column in grid 
    #store loss array in last column of grid 
    grid['loss'] = loss_array


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

#make a plot of the losses
plt.plot(loss_by_epoch_train)
plt.show()
plt.plot(loss_by_epoch_val)
plt.show()


plt.plot(history_model1.history['loss'],label='model1')
plt.plot(history_model2.history['loss'],label='model2')
plt.plot(history_model3.history['loss'],label='model3')
plt.plot(history_model4.history['loss'],label='model4')
plt.legend()
plt.show()