from sklearn.model_selection import GridSearchCV

def param_tune(estimator):
    param_grid = [{'batch_size':[], 'num_hid_layers':[], 'num_hid_neurons':[], 'hid_activation':['sigmoid', 'tanh','relu','elu','leaky relu','elu'], 'optimizer':['plain SGD','momentum','nesterov','adagrad','rmsprop','adam','learning rate scheduling']}]

    grid_search = GridSearchCV(estimator, param_grid,cv = 5, )

    return grid_search.best_params_