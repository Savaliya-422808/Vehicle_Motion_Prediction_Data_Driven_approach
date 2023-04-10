import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np


def FCN_keras_model(xTrain, xTest, yTrain, yTest):
    # flatten the input and output
    n_input = np.shape(xTrain)[1] * np.shape(xTrain)[2]
    xTrain = np.reshape(xTrain, (np.shape(xTrain)[0], n_input))
    n_output = np.shape(yTrain)[1] * np.shape(yTrain)[2]
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], n_output))

    xTest = np.reshape(xTest, (np.shape(xTest)[0], n_input))
    yTest = np.reshape(yTest, (np.shape(yTest)[0], n_output))

    # define multi-layer perceptron model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(n_input,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss='mse')

    # fit the keras model on the dataset
    history =model.fit(xTrain, yTrain, epochs=100, validation_data=(xTest, yTest))

    # save the trained model
    filepath = 'FCN_model.h5'
    model.save(filepath)

    # load trained model
    model = keras.models.load_model('FCN_model.h5')

    # Training and validation loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    return model