import numpy as np 
import tensorflow as tf
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.python.keras import regularizers
from sklearn.metrics import classification_report

"""
    Function to create model.

    @param training_X: training data
    @param training_y_encoded: one-hot encoded training labels
    @return bidrectional LSTM model
"""
def define_BiLSTM_model(training_X, training_y_encoded):
    n_timesteps, n_features, n_outputs = training_X.shape[1], training_X.shape[2], training_y_encoded.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(units = 125), input_shape = (n_timesteps, n_features)))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units = 125, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc', f1, precision_measure, recall_measure])
    return model

"""
    Save the model as a yaml file; and save the weights from the model.

    @param model: trained model to save
"""
def save_model(model):
    model_yaml = model.to_yaml()
    with open('model.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)
    # Serialize weights to HDF5
    model.save_weights('model.h5')
    print('Saved model to disk.')

"""
    Load the saved model.

    @param model_file_name: name of file that model was saved in
    @param weight_file_name: name of file that weights were saved in
    @return loaded model
"""
def load_model(model_file_name, weight_file_name):
    yaml_file = open(model_file_name, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # Load weights into new model
    loaded_model.load_weights(weight_file_name)
    print('Loaded model from disk.')
    return loaded_model

"""
    Performs a search of the most optimal hyperparameters.

    @param training_X: training data
    @param training_y_encoded: one-hot encoded training labels
    @param val_X: validation data
    @param val_y_encoded: one-hot encoded validation label
    @param dropout_grid: array of dropout values (0.1, 0.5, 0.9)
    @param l1_grid: array of L1 regularization values (2**-5, 2**-6, 2**-7, 2**-8)
    @param l2_grid: array of L2 regularization values (2**-5, 2**-6, 2**-7, 2**-8)
    @return best accuracy value, history of best model, index of best hyperparameter values
"""
def hyperparameter_search(training_X, training_y_encoded, val_X, val_y_encoded, dropout_grid, l1_grid, l2_grid):
    tot = len(dropout_grid) * len(l1_grid) * len(l2_grid)
    # Variables for the best result
    scores = []
    best_history = [] # Placeholder
    best_ind = 0
    best_acc = 0
    # Loop through each combination
    pos = 0
    for ii in dropout_grid:
        for jj in l1_grid:
            for kk in l2_grid:
                pos = pos + 1
                print("Fitting the ", pos, "/", tot , " model")
                # Define the model
                curr_model = define_LSTM_model(ii, jj, kk, training_X, training_y_encoded)
                
                # Train the model
                curr_history = curr_model.fit(training_X, training_y_encoded, epochs = 10, validation_data = (val_X, val_y_encoded), verbose = 0)
                curr_acc = st.mean(curr_history.history['val_acc'][5:10])
                            
                # Get prediction report
                y_pred = curr_model.predict(val_X, batch_size=64, verbose=0)
                y_pred_bool = np.argmax(y_pred, axis=1)
                scores.append(classification_report(val_y, y_pred_bool))
                
                # Save the best result
                if best_acc < curr_acc:
                    best_acc = curr_acc
                    best_ind = pos - 1
                    best_history = curr_history
    
    return best_acc, best_history, best_ind