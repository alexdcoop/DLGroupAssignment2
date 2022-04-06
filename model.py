#This is the final model file
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Read in data
DATA = pd.read_csv('pricing.csv')

# Split into X and Y
Y = DATA.pop('quantity')
X = np.array(np.column_stack((DATA['sku'],DATA['price'],DATA['order'],DATA['duration'],DATA['category'])))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


def createNeuralNet(batchSize, learning_rate, numHiddenLayers, numHiddenNeurons, hiddenActivationFunction, optimizer, epochs):
    inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
    hidden = tf.keras.layers.Dense(units=numHiddenNeurons, activation=hiddenActivationFunction)(inputs)
    
    for layer in numHiddenLayers - 1:
        hidden = tf.keras.layers.Dense(units=numHiddenNeurons, activation=hiddenActivationFunction)(hidden)

    output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden)
    
    #Create model 
    model = tf.keras.Model(inputs = inputs, outputs = output)

    #Compile model
    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

    #return model
    return model
    
#Make Grid of all possible parameters
param_grid = [{'batch_size':[100000, 50000, 10000, 5000, 1000], 'learning_rate':[.001,.01,.1,.2,.3], 'num_hid_layers':[5,4,3,2,1], 'num_hid_neurons':[200, 150, 125, 100, 75, 50, 25, 10, 5], 'hid_activation':['sigmoid', 'tanh','relu','elu','leaky relu','elu'], 'optimizer':['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']}]

