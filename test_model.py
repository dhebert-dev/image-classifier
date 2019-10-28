from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt

#import math functions
import math

from keras_preprocessing.image import img_to_array, load_img
from matplotlib import pyplot
from vis.losses import ActivationMaximization
from keras.layers.convolutional import Conv2D

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0

y_test = utils.to_categorical(y_test)

model = load_model('1024_model.h5')

model.load_weights('weights.DD.1024.50-0.70.hdf5')

results = model.evaluate(x=x_test, y=y_test)


def run_test():
    for i in range(610, 800):

        results = model.evaluate(x=x_test, y=y_test)
    
        test_image_data = np.asarray([x_test[i]])
        accuracy_data = np.asarray([y_test[i]])
    
        prediction = model.predict(x=test_image_data)
    
        np.set_printoptions(suppress=True)
    
        max_index = np.argmax(prediction)
    
        print("Prediction:", labels[max_index])
    
        plt.imshow(x_test[i])
        plt.show()
        
        
#print("Accuracy:", results[1])
accuracy = results[1]

