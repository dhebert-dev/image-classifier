# import CIFAR10 data
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

# import keras utils
import keras.utils as utils

# import Sequential modeling
from keras.models import Sequential

# import model layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# import maxnorm
from keras.constraints import maxnorm

# import optimizer
from keras.optimizers import SGD

# import h5py
import h5py

# import load model
from keras.models import load_model

from vis.losses import ActivationMaximization

########################################## END IMPORTS #################################################################
########################################## INITIALIZE ##################################################################

# load cifar10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# format x training and test data to float32 and divide by 255.0
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# load y training and test data into categories
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# create labels array
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

########################################## END INITIALIZE ##############################################################
######################################### SEQUENTIAL MODEL #############################################################

# initialize the model
model = Sequential()

# add first convolutional layer - Conv2d
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same',
                 kernel_constraint=maxnorm(3)))

# add convolutional layer - MaxPooling2d
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten features
model.add(Flatten())

# add convolutional layer - Dense
model.add(Dense(units=1024, activation='relu', kernel_constraint=maxnorm(3)))

# add convolutional layer - Dense
model.add(Dense(units=1024, activation='relu', kernel_constraint=maxnorm(3)))

# add convolutional layer - Dropout
model.add(Dropout(rate=0.5))

# add convolutional layer - Second Dense Layer
model.add(Dense(units=10, activation='softmax'))

#model.load_weights('weights.DD.1024.50-0.70.hdf5')
#print("Current weights loaded")
######################################## END SEQUENTIAL MODEL ##########################################################
########################################### COMPILER ###################################################################

model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

########################################## END COMPILER ################################################################
########################################## INITIAL TRAINING ############################################################

filepath = 'weights.DD.1024.{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='max', period=10)

callbacks_list = [checkpoint]

model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=50, batch_size=32, shuffle=True, callbacks=callbacks_list)

model.save('1024_model.h5')
