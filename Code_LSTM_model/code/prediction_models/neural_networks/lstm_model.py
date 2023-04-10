import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
#from keras.layers import Dropout

def my_lstm_model(xTrain, xTest, yTrain, yTest):
    
    #data = data_input
    #model = 0
    
    # flatten the input and output
    
    n_input = np.shape(xTrain)[1] * np.shape(xTrain)[2]
    xTrain = np.reshape(xTrain, (np.shape(xTrain)[0], n_input,1))
    n_output = np.shape(yTrain)[1] * np.shape(yTrain)[2]
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], n_output))

    xTest = np.reshape(xTest, (np.shape(xTest)[0], n_input,1))
    yTest = np.reshape(yTest, (np.shape(yTest)[0], n_output))
    
    # design network
    # define multi-layer perceptron model
    
    model = keras.Sequential()    
    model.add(LSTM(128, return_sequences=True,
               input_shape=(n_input,1)))
    model.add(LSTM(64, return_sequences=True)) 
    model.add(LSTM(32))  
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_output))
 
    model.compile(loss='mae', optimizer='adam')
   # callback =  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(xTrain, yTrain, epochs=50, batch_size=64, 
              validation_data=(xTest, yTest), verbose=1, shuffle=False)
    
    # fit network
    #history = model.fit(xTrain, yTrain, epochs=50, batch_size=72, validation_data=(xTest, yTest), verbose=2,
                        #shuffle=False)
                        
     # save the trained model
    filepath = 'LSTM_model.h5'
    model.save(filepath)

    # load trained model
    model = keras.models.load_model('LSTM_model.h5')

    # Training and validation loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    return model                    
    
    """
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    """
    return model









