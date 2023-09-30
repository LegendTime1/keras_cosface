import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler

from archs import CosFace  # Import the CosFace layer from your custom archs module

def main():
    # dataset
    (X, y), (X_test, y_test) = mnist.load_data()

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
    y_ohe = keras.utils.to_categorical(y, 10)
    y_ohe_test = keras.utils.to_categorical(y_test, 10)

    # feature extraction
    cosface_model = load_model('models/mnist_vgg8_cosface_3d/model.hdf5', custom_objects={'CosFace': CosFace})
    cosface_model = Model(inputs=cosface_model.input[0], outputs=cosface_model.layers[-3].output)
    cosface_features = cosface_model.predict(X_test, verbose=1)
    cosface_features /= np.linalg.norm(cosface_features, axis=1, keepdims=True)

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    for c in range(len(np.unique(y_test))):
        ax.plot(cosface_features[y_test==c, 0], cosface_features[y_test==c, 1], cosface_features[y_test==c, 2], '.', alpha=0.1)

    # Set a title for the plot
    ax.set_title('CosFace')

    # Save the plot as an image (e.g., PNG)
    fig.savefig('cosface_3d_plot.png')

if __name__ == '__main__':
    main()
