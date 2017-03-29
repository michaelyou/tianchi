import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import Dense, TimeDistributed, Dropout, Flatten, Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D

# tensorflow is different from theano
K.set_image_data_format('channels_first')

data = []
with open('data_sample.txt') as f:
    for line in f:
        data.append(line.split(','))

train_labels = np.array([d[1] for d in data]).astype('float32')
train_maps = np.array([d[2].split() for d in data]).astype('float32')
train_maps = train_maps.reshape(train_maps.shape[0], 15, 4, 101, 101)


def train_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(50, kernel_size=(5, 5), padding='valid'), input_shape=list(train_maps.shape)[1:]))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Conv2D(40, kernel_size=(3, 3), padding='valid')))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Flatten()))
    model.add(Activation('relu'))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

model = train_model()
model.fit(train_maps, train_labels, epochs=10, batch_size=1, verbose=1)
