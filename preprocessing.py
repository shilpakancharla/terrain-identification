import random
import numpy as np
import pandas as pd
import tensorflow as tf
import statistics as st
from scipy import stats
from sklearn.utils import class_weight
from sklearn.preprocessing import RobustScaler, OneHotEncoder

"""
    Based on the training data given, we are able to extract 7 attributes:
    1. x accelerometer measurement
    2. y accelerometer measurement
    3. z accelerometer measurement
    4. x gyroscope measurement
    5. y gyroscope measurement
    6. z gyroscope measurement
    7. time stamp for accelerometer and gyroscope measures
    
    We start by creating a dataframe using the csv files provided for readability.
    
    @param x_file: contains the xyz accelerometers and xyz gyroscope measures from the lower limb
    @param x_time_file: contain the time stamps for the accelerometer and gyroscope measures
    @return dataframe of 7 attributes mentioned
"""
def create_dataframe_X(x_file, x_time_file):
    df1 = pd.read_csv(x_file, sep = ',', names = ['X_acc', 'Y_acc', 'Z_acc', 'X_gyr', 'Y_gyr', 'Z_gyr'])
    df1 = scale_data(df1, ['X_acc', 'Y_acc', 'Z_acc', 'X_gyr', 'Y_gyr', 'Z_gyr'])
    df2 = pd.read_csv(x_time_file, names = ['Time stamp'])
    frames = [df1, df2]
    result = pd.concat(frames, axis = 1)
    return result

"""
    Scale the values of X to make it robust to outliers.
    
    @param df: input dataframe
    @param columns: columns to scale
    @return scaled dataframe
"""
def scale_data(df, columns):
    scaler = RobustScaler()
    scaler = scaler.fit(df[columns])
    df.loc[:, columns] = scaler.transform(df[columns])
    return df
    
"""
    We have both the labels and the time stamps for the labels. We create a dataframe from these for
    readability.
    
    @param y_file: contain the labels: 
        (0) indicates standing or walking in solid ground, 
        (1) indicates going down the stairs, 
        (2) indicates going up the stairs, and 
        (3) indicates walking on grass
    @param y_time_file: contain the time stamps for the labels
    @return dataframe of labels and time stamps
""" 
def create_dataframe_Y(y_file, y_time_file):
    df1 = pd.read_csv(y_file, names = ['Label'])
    df2 = pd.read_csv(y_time_file, names = ['Time stamp'])
    frames = [df1, df2]
    result = pd.concat(frames, axis = 1)
    return result
    
"""
    We take the outputs of create_dataframe_X and create_dataframe_Y. In order to combine both of these
    dataframes, we need look at the time intervals present for when the labels were assigned. We down-sample
    the X to the shape of the y.
    
    @param x_frame: dataframe from create_dataframe_X
    @param y_frame: dataframe from create_dataframe_Y
    @return dataframe with 9 columns (8 attributes and 1 label)
"""
def combine_frames(x_frame, y_frame):
    # Change each dataframe column to a list for iterations
    time_stamp_y = y_frame['Time stamp'].tolist()
    time_stamp_x = x_frame['Time stamp'].tolist()
    
    x_range = [] # Empty list to append data points to
    x_random_row = 0 # Initializing variable to hold randomly selected row instance
    refs = []
    count = 0
    for i in range(0, len(time_stamp_y)):
        while (time_stamp_x[count] <= time_stamp_y[i]) and (count <= len(time_stamp_x)):
            x_range.append(time_stamp_x.index(time_stamp_x[count]))
            count += 1
        x_random_row = random.choice(x_range) # Pick a random value
        refs.append(x_random_row) # Keep record of selected rows
        x_range.clear() # Clear the cache
        continue
    
    # Create a new dataframe based on the refs collected - should be roughly the same length as the y_frame
    entries = []
    for item in refs:
        entry = x_frame.iloc[item]
        entries.append(entry)
    
    found_df = pd.concat(entries, axis = 1)
    found_df = found_df.transpose()
    
    # Combine found_df with y_frame for downsampling
    found_df = found_df.reset_index()
    found_df = found_df.drop(['index'], axis = 1)
    found_df = found_df.drop(['Time stamp'], axis = 1)
    combined_frame = pd.concat([found_df, y_frame], axis = 1)
    return combined_frame

"""
    Takes in the sequential X and y and creates windows of time-series data.
    
    @param X: input data
    @param y: label data
    @param time_steps: determines size of window
    @param step: incremental value that window will slide over
    @return time series of X and y data
"""
def mode_labels(X, y, time_steps, step):
    X_values = []
    y_values = []
    for i in range(0, len(X) - time_steps, step):
        value = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: (i + time_steps)]
        X_values.append(value)
        y_values.append(stats.mode(labels)[0][0])
    return np.array(X_values), np.array(y_values).reshape(-1, 1)

"""
    Generating data frames from training data.
    
    @param X_file: list of input X files
    @param X_t_file: list of input X_time files
    @param y_file: list of input y files
    @param y_t file: list of y_time files
    @return stacked window of instances across all training files, stack window of labels across all label files
"""
def generate_data(X_file, X_t_file, y_file, y_t_file):
    all_X = []
    all_y = []
    for item_X, item_X_t, item_y, item_y_t in zip(X_file, X_t_file, y_file, y_t_file):
        df_x = create_dataframe_X(item_X, item_X_t)
        df_y = create_dataframe_Y(item_y, item_y_t)
        combined_frame = combine_frames(df_x, df_y)
        X_temp = combined_frame[['X_acc', 'Y_acc', 'Z_acc', 'X_gyr', 'Y_gyr', 'Z_gyr']]
        y_temp = combined_frame['Label']
        X, y = mode_labels(X_temp, y_temp, 30, 1)
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)

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