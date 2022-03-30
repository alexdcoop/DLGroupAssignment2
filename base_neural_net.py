from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import os

os.chdir(r'C:\Users\jwade\OneDrive - University of Tennessee\Masters Program\BZAN 554 - Deep Learning for Business\Group Assignment - 2')

# Read in data
DATA = pd.read_csv('pricing.csv')


# Split into X and Y
Y = DATA.pop('quantity')
X = DATA

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# Create grid
batchSize = [100000, 50000, 10000, 5000, 1000]
numHiddenLayers = [5,4,3,2,1]
numHiddenNeurons = []
hiddenActivationFunctions = ['sigmoid', 'tanh', 'relu', 'leaky relu', 'prelu', 'elu']
optimizers = ['plainSGD', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam', 'learning rate scheduling']

gridline = []
for i in range(5):
    gridline.append("")
grid = []
for i in range(5):
    grid.append(list(gridline))

def createNeuralNet(batchSize, numHiddenLayers, numHiddenNeurons, hiddenActivationFunction, optimizer):
    inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
    hidden = tf.keras.layers.Dense(units=250, activation="elu")(inputs)
    
    for units in [200, 150, 125, 100, 75, 50, 25, 10, 5]:
        hidden = tf.keras.layers.Dense(units=units, activation="elu")(hidden)

    output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden)


#Go through each row in grid and make neural net

#Get the most accurate neural net

