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
from livelossplot.keras import PlotLossesCallback
import tensorflow as tf


def generate_np_arrays(df):
    original_arrays = df.image
    images = []
    for sample in original_arrays:
        image = np.array(sample.split(), dtype="float32")
        image = image.reshape(48, 48)
        images.append(image)
    df["image"] = images
    return df


def main():
    img_size = 48
    batch_size = 64  # todo: hyperparameter
    datagen_train = ImageDataGenerator(horizontal_flip=True)  # todo: consider if necessary
    train_generator = datagen_train.flow_from_directory("train/", target_size=(img_size, img_size),
                                                        color_mode="grayscale", batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)

    val_generator = datagen_train.flow_from_directory("test/", target_size=(img_size, img_size), color_mode="grayscale",
                                                      batch_size=batch_size, class_mode='categorical', shuffle=True)

    model = Sequential()
    # conv1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # conv2
    model.add(Conv2D(128, (5, 5), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # conv3
    model.add(Conv2D(512, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # conv4
    model.add(Conv2D(512, (5, 5), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    """
    Train and Evaluate
    """
    epochs = 15
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = val_generator.n // val_generator.batch_size
    check_point = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', save_weights_only=True, mode='max',
                                  verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')
    call_backs = [PlotLossesCallback(), check_point, reduce_lr]
    history = model.fit(x=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=val_generator, validation_steps=validation_steps, callbacks=call_backs)


if __name__ == '__main__':
    main()
