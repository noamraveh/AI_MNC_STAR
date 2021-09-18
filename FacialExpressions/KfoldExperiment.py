import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from livelossplot import PlotLossesKeras
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
import tensorflow as tf
from sklearn.model_selection import KFold
import itertools


class ConvLayer:
    def __init__(self, filter, max_pooling, is_input=0):
        self.filter = filter
        self.max_pooling = max_pooling
        self.is_input = is_input


class FullyLayer:
    def __init__(self, dense):
        self.dense = dense


def architecture_experiment(X, y):
    kfold = KFold(n_splits=5, random_state=10, shuffle=True)

    # conv
    num_conv_layers = range(0, 3)  # 1,2,3
    num_filters = [2 ** i for i in range(5, 8)]  # 32,64,128
    dropout = 0.25
    kernel_size = (3, 3)
    max_pooling = {"option_1": {"pool_size": (2, 2), "strides": 2}, "option_2": {"pool_size": (3, 3), "strides": 3}}

    # fully connected
    num_fully_connected = range(1, 3)  # 1,2
    dense_vals = [2 ** i for i in range(5, 8)]  # 32,64,128

    # generate all conv layers options
    conv_layers_input_layer_all_options = []
    conv_layers_inner_layer_all_options = []
    for filter in num_filters:
        for key in max_pooling.keys():
            input_layer = ConvLayer(filter, max_pooling[key], 1)
            inner_layer = ConvLayer(filter, max_pooling[key])
            conv_layers_input_layer_all_options.append(input_layer)
            conv_layers_inner_layer_all_options.append(inner_layer)

    conv_options = []
    for input_layer in conv_layers_input_layer_all_options:
        for i in num_conv_layers:
            options = itertools.permutations(conv_layers_inner_layer_all_options, i)
            options = [list(option) for option in options]
            for option in options:
                layers = [input_layer]
                layers.extend(option)
                conv_options.append(layers)

    # generate all fully connected layers options:
    fully_connected_layer_all_options = []
    for dense in dense_vals:
        fully_layer = FullyLayer(dense)
        fully_connected_layer_all_options.append(fully_layer)

    fully_connected_options = []
    for i in num_fully_connected:
        options = list(itertools.permutations(fully_connected_layer_all_options, i))
        options = [list(option) for option in options]
        for option in options:
            fully_connected_options.append(option)

    batch_size = 64
    all_acc = []
    all_params = []
    model_number = 0
    for conv_option in conv_options:
        for fully_connected_option in fully_connected_options:
            all_params.append((conv_option, fully_connected_option))
            sum_acc = 0
            model = Sequential()
            for layer in conv_option:
                if layer.is_input:
                    model.add(Conv2D(layer.filter, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
                else:
                    model.add(Conv2D(layer.filter, kernel_size=(3, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(
                    MaxPooling2D(pool_size=layer.max_pooling["pool_size"], strides=layer.max_pooling["strides"]))
                model.add(Dropout(0.25))

            model.add(Flatten())

            for layer in fully_connected_option:
                model.add(Dense(layer.dense))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

            model.add(Dense(7, activation='softmax'))

            opt = Adam(learning_rate=0.0005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"model_number:{model_number}")
            model_number += 1
            # plot_model(model, to_file=f'model__{i}.png', show_shapes=True, show_layer_names=True)
            # Image(f'model__{i}.png', width=400, height=200)

            for train_group_idx, test_group_idx in kfold.split(X):
                x_train = X[train_group_idx]
                y_train = y[train_group_idx]
                x_val = X[test_group_idx]
                y_val = y[test_group_idx]

                epochs = 10
                steps_per_epoch = len(x_train) // batch_size
                steps_per_epoch = steps_per_epoch // 5
                validation_steps = len(x_val) // batch_size
                trained_cnn = model.fit(x=x_train, y=y_train, steps_per_epoch=steps_per_epoch, epochs=epochs)
                predicted_values = model.predict(x=x_val, batch_size=64)
                num_hits = sum(1 for pred, true_val in zip(predicted_values, y_val) if
                               np.argmax(pred, axis=0) == np.argmax(true_val, axis=0))
                cur_accuracy = num_hits / float(len(y_val))
                sum_acc += cur_accuracy
            avg_acc = sum_acc / float(5)
            all_acc.append(avg_acc)
    all_acc = np.array(all_acc)
    best_acc_idx = np.argmax(all_acc, axis=0)
    best_params = all_params[best_acc_idx]
    print(f"best accuracy: {all_acc[best_acc_idx]}")
    for i, conv in enumerate(best_params[0]):
        print(f"convolutional layer {i + 1}:")
        print(f"filters: {conv.filter}")
        print(f"pool_size: {conv.max_pooling['pool_size']}")
        print(f"strides: {conv.max_pooling['strides']}")
    for i, fully in enumerate(best_params[1]):
        print(f"dense layer {i + 1}:")
        print(f"units: {fully.dense}")


def main():
    img_size = 48
    batch_size = 64  # todo: hyperparameter
    datagen_train = ImageDataGenerator()  # todo: consider if necessary
    train_generator = datagen_train.flow_from_directory("/content/images/train/", target_size=(img_size, img_size),
                                                        color_mode="grayscale", batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)

    total_images = train_generator.n
    steps = total_images / batch_size
    X, y = [], []
    for i in range(round(steps)):
        a, b = train_generator.next()
        X.extend(a)
        y.extend(b)
    X = np.array(X)
    y = np.array(y)

    architecture_experiment(X, y)


if __name__ == '__main__':
    main()
