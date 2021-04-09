import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras import regularizers
from keras.optimizers import Adam
from keras_radam import RAdam 
from evaluation import f1, precision_measure, recall_measure

"""
    Function to create model. Required for KerasClassifier.
"""

def define_LSTM_model(dropout_rate = 0.1, l1_value = 2**-5, l2_value = 2**-5, training_X, training_y):
    n_timesteps, n_features, n_outputs = training_X.shape[1], training_X.shape[2], training_y.shape[1]
    initializer = tf.keras.initializers.Orthogonal()
    model = Sequential()
    model.add(LSTM(units = 125, kernel_initializer = initializer, kernel_regularizer = regularizers.l1_l2(l1 = l1_value, l2 = l2_value),
                    input_shape = (n_timesteps, n_features)))
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(units = 75, activation = 'tanh'))
    model.add(Dense(units = n_outputs, activation = 'softmax'))
    model.compile(RAdam(), loss = 'categorical_crossentropy', metrics = ['accuracy', f1, precision_measure, recall_measure])
    return model