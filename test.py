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
# with open('data_sample.txt') as f:
with open('data_new_train/CIKM2017_train/train.txt') as f:
    # for line in f:
    #    data.append(line.split(','))
    data = [next(f).split(',') for x in range(20)]

train_labels = np.array([d[1] for d in data]).astype('float32')
print(train_labels)
train_maps = np.array([d[2].split() for d in data]).astype('float32')
train_maps = train_maps.reshape(train_maps.shape[0], 15, 4, 101, 101)


def train_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(50, kernel_size=(5, 5), padding='valid'), input_shape=list(train_maps.shape)[1:]))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(3, 3), padding='valid')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(3, 3), padding='valid')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Conv2D(20, kernel_size=(3, 3), padding='valid')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Flatten()))
    # print(model.output_shape)
    model.add(Activation('relu'))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # print([layer.output for layer in model.layers])
    return model

model = train_model()
model.fit(train_maps, train_labels, epochs=5, batch_size=1, verbose=1)
results = model.predict(train_maps)
print(results)
