# Cats and Dogs Classifier
In this Machine Learning Model, we are implementing the CNN (Convolutional Neural Network)  to classify dogs and cats images.

![Python 3.7](https://img.shields.io/badge/python-v3.7-blue)    ![Problem Kaggle](https://img.shields.io/badge/Problem-Vision-blue.svg)     ![Problem Kaggle](https://img.shields.io/badge/Data-Kaggle-orange.svg)

## In this NN, we are implementing the CNN to classify dogs and cats images.
* For training the model, we have used this dataset From Kaggle > [Download](https://www.kaggle.com/ppleskov/cute-cats-and-dogs-from-pixabaycom)
* Due to upload size limit of github, we have uploaded trained model on Google Drive > [Download](https://drive.google.com/uc?export=download&id=1CLolexl8DJWLKRt70tJVZLCC3f3XY9zu)

## Dependencies
* Jupyter notebook
* Tensorflow 2.0
* Python 3.7
* Matplotlib
* numpy

## Layer Architecture
INPUT => [CN => ELU => BN => POOL => DO] x 2 => [FN => ELU => BN => DO] x 4 => FN => SIGMOID

```
from tensorflow import nn
from numpy import expand_dims
from numpy import array as arr
from numpy import float32
import os
import cv2
import matplotlib.pyplot as plt
```

## Network Parameters
* Activation: elu
* Output Activation: sigmoid
* BatchNormalization, For normalizing the activations
* For Regularization, we used: Dropout with 0.25
* Adam as optimizer
* Binary Cross Entropy as loss function

```
classifiers_cats_dog_model = keras.Sequential([
    keras.layers.Conv2D(64, (5,5), padding="same", activation=nn.elu, input_shape=(60,60,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (5,5), padding="same", activation=nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(1000, activation=nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(750, activation=nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(500, activation=nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(250, activation=nn.elu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation=nn.sigmoid)
])

opt_v = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

classifiers_cats_dog_model.compile(optimizer=opt_v, loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

```

## Data preprocessing/Data Augmentation
In this process, we load the training and validation data from directory and also apply some augmentation techniques to get the better results during training.

We applied these augmentations on images:
* Rotations
* width shift
* height shift
* image zoom
* Image Flipping (horizontal)

We used rescaling argument to normalize the image pixels.

In training data, we have used 31,764 images for two categories:
* Total cats images: 15,882
* Total dogs images: 15,882

In validation data, we have used 12,508 images for two categories:
* Total cats images: 6,254
* Total dogs images: 6,254

```
training_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=15, 
                                                                          width_shift_range=0.15,
                                                                         height_shift_range=0.15,
                                                                         zoom_range=0.15,
                                                                         horizontal_flip=True,
                                                                         fill_mode="nearest")

validation_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#training data
training_data_generated = training_data_generator.flow_from_directory(training_set, target_size=(60,60), 
                                                                     class_mode="binary", batch_size=256, shuffle=True,
                                                                     seed=42)

#validation data
validation_data_generated = validation_data_generator.flow_from_directory(validation_set, target_size=(60,60), 
                                                                          class_mode="binary", batch_size=256,
                                                                         shuffle=True, seed=42)
```
```
#Training Phase
classifiers_cats_dog_model.fit_generator(training_data_generated, steps_per_epoch=124, epochs=50, validation_data=validation_data_generated, validation_steps=48)
```

### Prediction of a Single Image

```
classifier = Model_loader() #Model _loader() is a class that will load the classifier and performs the required functions.
prediction = classifier.img_prediction(classifier.convert_img("maxresdefault (1).jpg"))
plt.figure(figsize=(10,5))
plt.text(100,100, prediction, color="red", fontsize=24, bbox=dict(facecolor="white", alpha=0.8))
plt.imshow(classifier.loaded_img)
plt.show()
```
![png](https://1.bp.blogspot.com/-CcQnzm7HowM/XhBmYwnpZ5I/AAAAAAAACw4/-JC-bJ6D6VUUdU8w-yYpk-rnn8JtYzJNQCLcBGAsYHQ/s1600/download.png)
