from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from numpy import expand_dims

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[0].output)

img = load_img('headshot2.jpg', target_size=(224, 224))

img = img_to_array(img)

img = expand_dims(img, axis=0)

img = preprocess_input(img)

feature_maps = model.predict(img)

square = 8
ix = 1

for _ in range(square):
    for _ in range(square):
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
        
pyplot.show()