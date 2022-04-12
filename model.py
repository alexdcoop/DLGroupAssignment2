#This is the final model file
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from random import sample
from sklearn.metrics import mean_absolute_error
from ann_visualizer.visualize import ann_viz


# Read in data
DATA = pd.read_csv('pricing.csv')

# Split into X and Y
Y = DATA.pop('quantity')
X = np.array(np.column_stack((DATA['sku'],DATA['price'],DATA['order'],DATA['duration'],DATA['category'])))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def createNeuralNet(learning_rate, num_hid_layers, num_hid_neurons, hid_activation, optimizer):
    shape = 1
    #Create inputs layer
    inputs_cat = tf.keras.layers.Input(shape=(shape,),name = 'in_cat')
    inputs_quant = tf.keras.layers.Input(shape=(shape,),name = 'in_quant')
    embedding_category = tf.keras.layers.Embedding(input_dim=32, output_dim=26, input_length=1,name = 'embedding_category')(inputs_cat)
    embedding_sku = tf.keras.layers.Embedding(input_dim=74999, output_dim=50000, input_length=1,name = 'embedding_sku')(inputs_quant)
    embedding = tf.keras.layers.Concatenate(name = 'concatenation_category')([embedding_category, embedding_sku])
    embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
    inputs_num = tf.keras.layers.Input(shape=(shape,),name = 'in_num')
    inputs = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
    
    hidden = tf.keras.layers.Dense(units=num_hid_neurons, activation=hid_activation)(inputs)
    
    for layer in range(num_hid_layers - 1):
        hidden = tf.keras.layers.Dense(units=num_hid_neurons, activation=hid_activation)(hidden)

    output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden)
    
    #Create model 
    model = tf.keras.Model(inputs = inputs, outputs = output)
    
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

    #Compile model
    model.compile(loss = 'mae', optimizer = optimizerCode)

    #return model
    return model
    
#Make Grid of all possible parameters
#param_grid = [{'learning_rate':[.001,.01,.1,.2,.3], 'num_hid_layers':[5,4,3,2,1], 'num_hid_neurons':[75, 50, 25, 10, 5], 'hid_activation':['sigmoid', 'tanh','relu','elu'], 'optimizer':['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']}]
glorot_sigm_init = keras.initializers.VarianceScaling(scale = 16, mode ='fan_avg', distribution = 'untruncated_normal')
he_avg_init = keras.initializers.VarianceScaling(scale = 2, mode = 'fan_avg', distribution = 'untruncated_normal')

learning_rate = [.001,.01,.1,.2,.3]
num_hid_layers = [5,4,3,2,1]
num_hid_neurons=[75, 50, 25, 10, 5]
hid_activation=['sigmoid', 'tanh','relu','elu']
optimizer=['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']
initializer=['glorot_normal', glorot_sigm_init, 'he_normal', he_avg_init]

#Generate all combinations
import itertools
a = [learning_rate,num_hid_layers,num_hid_neurons,hid_activation,optimizer, initializer]
allCombinations = list(itertools.product(*a))


def createModel(learning_rate, num_hid_layers, num_hid_neurons, hid_activation, optimizer, initializer):
    #Create model
    model = keras.Sequential()
    
    #Add input layer
    model.add(keras.Input(shape=(X.shape[1],)))

    #make initialiizer
    #if initializer == 'glorot_sigm_init':
    #    initializer = keras.initializers.VarianceScaling(scale = 16, mode ='fan_avg', distribution = 'untruncated_normal')
    #elif initializer == 'he_avg_init':
    #    initializer = keras.initializers.VarianceScaling(scale = 2, mode = 'fan_avg', distribution = 'untruncated_normal')
    
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
    model.compile(loss = 'mae', optimizer=optimizerCode, metrics=['accuracy'])
    
    return model



combinations = []
losses = []
epochs = 100
batchSize = 100000
numCombinations = 100

#use random subset
randomCombinations = sample(allCombinations,numCombinations)

for record in randomCombinations: 
    #Create Model
    model = createModel(learning_rate = record[0], num_hid_layers = record[1], num_hid_neurons = record[2], hid_activation = record[3], optimizer = record[4], initializer = record[5])
    
    lossList = []
    for _ in range(epochs):
        #fit model
        model.fit(X_train,Y_train,epochs=1,batch_size=batchSize, verbose=0)
        
        #predict
        yhat = model.predict(X_test)
        
        #compute loss (Keep 100 loss values)
        #manually compute loss
        #use built in tf mae loss function
        lossList.append(mean_absolute_error(Y_test, yhat)) #append to list
    
    #append to combinations and lossess
    combinations.append(record)
    losses.append(lossList)
     
#best model
best = np.argmin(losses)   
bestmodel = combinations[best]

#plots   
import matplotlib.pyplot as plt    
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()
    
ann_viz(bestmodel, title="Our Neural Network", filename="dlproj2.gv", view = True) #should produce a visual of the neural net
        

# #Batch sizes
# {'batch_size':[100000, 50000, 10000, 5000, 1000]}

# model = KerasRegressor(build_fn=createModel, epochs=10, batch_size=100000, verbose=0)

# #Grid search
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# gridResult = grid.fit(X,Y)

# # summarize results
# print("Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))
# means = gridResult.cv_results_['mean_test_score']
# stds = gridResult.cv_results_['std_test_score']
# params = gridResult.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    
# #Random search
#rnd_search = RandomizedSearchCV(model, param_grid, n_iter =20, cv=3)
#rnd_search.fit(X_train,Y_train)

# # summarize results
# print("Best: %f using %s" % (rnd_search.best_score_, rnd_search.best_params_))
# means = rnd_search.cv_results_['mean_test_score']
# stds = rnd_search.cv_results_['std_test_score']
# params = rnd_search.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    
    
# model = createModel(.2,3,5,'tanh','rmsprop','glorot_normal')

# model.fit(X_train, Y_train,epochs=10,batch_size=10000)
# model.summary()

# y = model.predict(X_test)
