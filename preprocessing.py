import random
import numpy as np
import pandas as pd
import tensorflow as tf
import statistics as st
from scipy import stats
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder

"""
    Based on the training data given, we are able to extract 7 attributes:
    1. x accelerometer measurement
    2. y accelerometer measurement
    3. z accelerometer measurement
    4. x gyroscope measurement
    5. y gyroscope measurement
    6. z gyroscope measurement
    
    We start by creating a dataframe using the csv files provided for readability. Since the sampling rates for the input data are
    at 40 Hz while the sampling rate for the labels is 10 Hz, we extrapolate 4 labels to match the sampling rate of the input data.
    We consider this up sampling.
    
    @param x_file: contains the xyz accelerometers and xyz gyroscope measures from the lower limb
    @param y_file: contain the labels for the accelerometer and gyroscope measures
    @return dataframe of input and dataframe of extrapolated labels to match number of inputs
"""
def upsampling(X_file, y_file):
    # Read in both CSV files
    df_X = pd.read_csv(X_file)
    df_y = pd.read_csv(y_file)
    
    extrapolated_labels = []
    # Iterate through every item (row) of the y labels
    for label in df_y.iterrows():
        # Get the label, add it four times
        extrapolated_labels += [label[1][0]] * 4
    
    # Create a dataframe, not series, so that you get the shapes - keep consistent
    extrapolated_labels_df = pd.DataFrame(extrapolated_labels)
    # X and extrapolated labels may not be the same length - account for differences here
    difference = df_X.shape[0] - extrapolated_labels_df.shape[0]
    df_X = df_X.iloc[:-difference,:]
    
    return df_X, extrapolated_labels_df

"""
    Scale the values of X to make it robust to outliers. We convert our dataframe to a numpy array as this makes it easier to create a
    three-dimensional array of inputs to pass into our model.
    
    @param df: input dataframe
    @param columns: columns to scale
    @return scaled array of input values
"""
def scale_data(df, columns):
    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df.loc[:, columns] = scaler.transform(df[columns].to_numpy())
    return df # Although called a dataframe, this is an array that gets returned

"""
    Takes in the sequential X and y and creates windows of time-series data. We take the mode of the labels to create each instances
    of the y_values.
    
    @param X: input data (dataframe structure)
    @param y: label data (dataframe structure)
    @param time_steps: determines size of window
    @param step_size: incremental value that window will slide over
    @return time series of X and y data in numpy.ndarrays
"""
def mode_labels(X, y, time_step, step_size):
    X_values = []
    y_values = []
    for i in range(0, len(X) - time_step, step_size):
        value = X.iloc[i:(i + time_step)].values
        labels = y.iloc[i:(i + time_step)]
        X_values.append(value)
        y_values.append(stats.mode(labels)[0][0])
    return np.array(X_values), np.array(y_values).reshape(-1, 1)

"""
    Using the sliding window technique to generate time series data of shape (number of samples, window size, number of features). We
    combine all the data passed in as lists of X and y files to create this.
    
    @param X_files: list of input X files
    @param y_files: list of input y files files
    @param time_steps: determines size of window
    @param step_size: incremental value that window will slide over
    @return stacked window of instances across all training files, stack window of labels across all label files
"""
def create_time_series_data(X_files, y_files, time_step, step_size):
    aggregate_X = []
    aggregate_y = []
    for i in range(len(y_files)):
        X, y = upsampling(X_files[i], y_files[i])
        X = scale_data(X, list(X.columns.values))
        X, y = mode_labels(X, y, time_step, step_size)
        aggregate_X.append(X)
        aggregate_y.append(y)
    return np.concatenate(aggregate_X), np.concatenate(aggregate_y)

"""
    We handle the data imbalance by assign higher weights to minority classes.

    @param training_X: training X data
    @param training_y: labels for training data
    @return dictionary of labels as key and weights as values
"""
def get_label_weights(training_X, training_y):
    label_weights = class_weight.compute_class_weight('balanced', np.unique(training_y), training_y.ravel())
    label_weights = {i:label_weights[i] for i in range(len(label_weights))}
    return label_weights

"""
    Perform one-hot encoding of the data to feed into our model.

    @param labels: labels of the training data
    @return one-hot encoded version of the labels
"""
def one_hot_encoding(labels):
    encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
    encoder = encoder.fit(labels)
    training_y_encoded = encoder.transform(labels)
    return training_y_encoded