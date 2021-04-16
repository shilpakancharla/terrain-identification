# Terrain Identification from Time Series Data

## Background

Humans naturally develop walking capability that is energy efficient, stable, environment adaptive, and robust. Lower limb amputations, unfortunately, disrupt this ability; individuals with lower limb amputations usually depend on prosthetic devices to restore the basic walking function. Lower-limb robotic prosthetics can benefit from context awareness to provide enhanced comfort and safety to the amputee. In this work, we aim to develop a terrain identification system based on inertial measurement units IMU streams collected from the lower limb. The system for a prosthetic leg uses visual and inertial sensors though, but we are willing to observe if terrain identification without the visual data is viable. With such information, the control of a robotized prosthetic leg can be adapted to changes in its surrounding.

## Task

This is a classification task to find different terrains from time series data. The idea is to train a neural network using given data to classify which terrain an unknown data represents. We will use **F1 score** as the evaluation metric of this project.

## Description of data:

* **"_x"** files contain the xyz accelerometers and xyz gyroscope measures from the lower limb.
* **"_x_time"** files contain the time stamps for the accelerometer and gyroscope measures. The units are in seconds and the sampling rate is 40 Hz.
* **"_y"** files contain the labels. (0) indicates standing or walking in solid ground, (1) indicates going down the stairs, (2) indicates going up the stairs, and (3) indicates walking on grass.
* **"_y_time"** files contain the time stamps for the labels. The units are in seconds and the sampling rates is 10 Hz. 

## Description of Code

**This project requires `tensorflow 1.8.0` and `keras 2.4.3`.**

* `preprocessing.py`: This script contains functions to do upsampling of our training data and create the time series array structure that we pass into our final bidirectional LSTM. Moreoever, it contains methods to gather the weights of each of the labels and perform one-hot encoding to the labels to pass into our model. 
* `evaluation.py`: This script contains functions that calculate the precision, recall, and F1-score. Moreoever, we also have a function that generates plots of the training and validation classifiation measures (i.e., loss in categorical cross-entropy, accuracy, precision, recall, and the F1-score). Finally, this script also contains functions which do the post-processing steps required to create the final predictions on the hidden test set. Namely, `create_dataset` will be used to create the time-series structure that we initially passed into our model, and `get_majority` reduces the size of the output by 4 (becuase the input data is sampled at 40 Hz, and the output time stamps show that it is sampled every 10 Hz).
* `model.py`: This script contains our final bidirectional LSTM structure. The method `hyperparameter_search` demonstrates how we initially gathered values for the dropout rate, L1 regularization multiplier, and L2 regularization multipler. Finally, there are also functions to save and load our model once it has been run. 
* `Proj_C2_BiLSTM.ipynb`: In this notebook, we load our data, run the model (bidirectional LSTM), produce various plots shown in our paper, and generate the final predictions.
* `Proj_C1_CNN.ipynb`: This notebook was our initial experiment in part one of this competition, in which we use a 1D-CNN.

## Loading the model

The model and the weights associated with it have already been saved here as `model.yaml` and `model.h5`, respecively. You can use the `load_model` function from `model.py` to pass in these two file names in order to run the model if needed. 
