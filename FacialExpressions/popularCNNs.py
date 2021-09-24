from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np


# ResNet
def ResNetModel(train, validation):
    resNet = ResNet50(include_top=False, weights=None, input_shape=(48, 48, 1))
    output = resNet.layers[-1].output
    output = keras.layers.Flatten()(output)
    resNet = Model(resNet.input, output)
    for layer in resNet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resNet)
    model.add(Dense(512, activation='relu', input_dim=(48, 48, 1)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    epochs = 20
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    history = model.fit_generator(train, epochs=epochs, validation_data=validation, callbacks=[callback])


def AlexNetModel():
    model = tf.keras.models.Sequential([
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
        BatchNormalization(),
        MaxPooling2D(2, strides=(2, 2)),
        Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(7, activation='softmax')
    ])

    image_generator = ImageDataGenerator(rescale=1. / 255)
    # training_data for model training - AlexNet requires 227x227 images..
    train_data_gen = image_generator.flow_from_directory(directory="train/",
                                                         batch_size=64,
                                                         shuffle=True,
                                                         target_size=(227, 227),
                                                         # Resizing the raw dataset
                                                         class_mode='categorical')

    val_data_gen = image_generator.flow_from_directory(directory="test/",
                                                       batch_size=64,
                                                       shuffle=True,
                                                       target_size=(227, 227),
                                                       # Resizing the raw dataset
                                                       class_mode='categorical')

    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 20
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    history = model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen, callbacks=[callback])


def LeNetModel(train, validation):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 20
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    history = model.fit(train, epochs=epochs, validation_data=validation, callbacks=[callback])


def main():
    img_size = 48

    datagen_train = ImageDataGenerator()
    train_generator = datagen_train.flow_from_directory("train/", target_size=(img_size, img_size),
                                                        color_mode="grayscale",
                                                        class_mode='categorical', shuffle=True)

    val_generator = datagen_train.flow_from_directory("test/", target_size=(img_size, img_size),
                                                      color_mode="grayscale",
                                                      class_mode='categorical', shuffle=True)

    # ResNetModel(train_generator, val_generator)
    # AlexNetModel()
    # LeNetModel(train_generator, val_generator)


if __name__ == '__main__':
    main()
