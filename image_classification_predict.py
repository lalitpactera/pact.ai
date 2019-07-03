from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K
import numpy as np
from keras.applications.vgg19 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def image_model_predict(img_path):
    filepath="_model_weights.h5"
    height=224
    width=224
    load_pretrained=False
    freeze_layers_from='base_model'
    img_shape=(224, 224,3)
    adam = Adam(lr=0.00001)
    lossfn = 'binary_crossentropy'

    base_model = ResNet50(include_top=False, weights=None,
                input_tensor=None, input_shape=img_shape)

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(2, activation= 'softmax', name='fc1000')(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights(filepath)
    model.compile(adam, loss=lossfn, metrics=['accuracy'])
    image1 = image.load_img(img_path, target_size=(224, 224))
    image1 = image.img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    image1 = preprocess_input(image1)
    probabilities = model.predict(image1)
    op = np.argmax(probabilities, axis=1)
    #print(op)
    if op == 0:
        return 'Normal'
    if op == 1:
        return 'Abnormal'  
        