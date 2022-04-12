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
epochs = 10
batchSize = 100000
numCombinations = 50

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
minLosses = [np.min(loss) for loss in losses]
bestIndex = np.argmin(minLosses)
bestModel = 
best = np.argmin(losses)   
bestmodel = combinations[best]

#plots   
import matplotlib.pyplot as plt
for i in range(len(losses)):
    plt.plot(losses[i]) 
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()
    
#ann_viz(bestmodel, title="Our Neural Network", filename="dlproj2.gv", view = True) #should produce a visual of the neural net

