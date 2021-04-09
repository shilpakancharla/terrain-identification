import matplotlib.pyplot as plt
from keras import backend as K

"""
    Calculate recall from predicted and actual values.

    @param y_true: actual y values
    @param y_pred: predicted y values
"""
def recall_measure(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

"""
    Calculate precision from predicted and actual values.

    @param y_true: actual y values
    @param y_pred: predicted y values
"""
def precision_measure(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

"""
    Calculate F1-score from predicted and actual values.

    @param y_true: actual y values
    @param y_pred: predicted y values
"""
def f1(y_true, y_pred):
    precision = precision_measure(y_true, y_pred)
    recall = recall_measure(y_true, y_pred)
    return 2 * ((precision * recall)/(precision + recall + K.epsilon()))

"""
    Defining a function for plotting training and validation learning curves.

    @param history: model history with all the metric measures
"""
def plot_history(history):
	# Plot loss
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # Plot accuracy
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='red', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # Plot F1
    plt.title('F1-Score')
    plt.plot(history.history['f1'], color='blue', label='train')
    plt.plot(history.history['val_f1'], color='red', label='test')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # Plot precision
    plt.title('Precision')
    plt.plot(history.history['precision_measure'], color='blue', label='train')
    plt.plot(history.history['val_precision_measure'], color='red', label='test')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # Plot recall
    plt.title('Recall')
    plt.plot(history.history['recall_measure'], color='blue', label='train')
    plt.plot(history.history['val_recall_measure'], color='red', label='test')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()