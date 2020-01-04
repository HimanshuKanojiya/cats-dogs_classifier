# cats and dogs classifier
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
