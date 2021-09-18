import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math


class ArchitectureExperiments:
    def __init__(self):
        self.list_of_conv_layers = [1, 2, 3, 4]
        self.list_of_fully_connected_layers = [1, 2, 3]
        self.list_of_filters = [32, 64, 128, 256, 512]
        self.list_of_kernel_sizes = [(3, 3), (5, 5)]
        self.list_of_units = [32, 64, 128, 256, 512]
        self.list_of_dropouts = [i * 0.1 for i in range(7)]
        self.list_of_lr = [0.001, 0.01, 0.1, 1]
        self.list_of_batch_sizes = [32, 64, 128]
        self.list_of_max_pooling = [2, 3]

        self.img_size = 48
        self.default_batch_size = 64
        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.default_batch_size,
                                                            class_mode='categorical', shuffle=True)

        total_images = train_generator.n
        steps = total_images // self.default_batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        self.X = np.array(X)
        self.y = np.array(y)

    def number_of_conv_layers(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_conv_layers)
        models_names = []
        for i in range(len(self.list_of_conv_layers)):
            models_names.append(f"{i + 1}C")
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.25))
            if i > 0:
                model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))
            if i > 1:
                model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))
            if i > 2:
                model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))

            model.add(Flatten())

            model.add(Dense(64))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

            model.add(Dense(7, activation="softmax"))
            opt = Adam(learning_rate=0.0005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Number of Convolutional Layers")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def number_of_fully_connected_layers(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_fully_connected_layers)
        models_names = []
        for i in range(len(self.list_of_fully_connected_layers)):
            models_names.append(f"{i + 1}F")
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.25))

            model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.25))

            model.add(Flatten())

            model.add(Dense(32))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

            if i > 0:
                model.add(Dense(64))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

            if i > 1:
                model.add(Dense(128))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

            model.add(Dense(7, activation="softmax"))
            opt = Adam(learning_rate=0.0005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Number of Fully Connected Layers")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def filters(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_filters)  # todo: calc num of permutations
        models_names = []
        for i in range(len(self.list_of_filters)):
            for j in range(i, len(self.list_of_filters)):
                models_names.append(f"{self.list_of_filters[i]},{self.list_of_filters[j]}")
                model = Sequential()
                model.add(Conv2D(self.list_of_filters[i], kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))

                model.add(Conv2D(self.list_of_filters[j], kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))

                model.add(Flatten())

                model.add(Dense(64))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

                model.add(Dense(7, activation="softmax"))
                opt = Adam(learning_rate=0.0005)
                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                epochs = 20
                callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
                cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                        validation_data=(X_val, y_val), callbacks=[callback])
                results[i] = cur_history

            plt.figure()
            for result in results:
                plt.plot(result.history["val_accuracy"])
            plt.title("Model Accuracy by Filters of Convolutional Layers")
            plt.xlabel("epochs")
            plt.ylabel("validation Accuracy")
            plt.legend(models_names, loc='upper left')
            plt.show()
            for name, result in zip(models_names, results):
                print(f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def dense_units(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_units)  # todo: calc num of permutations
        models_names = []
        for i in range(len(self.list_of_units)):
            for j in range(i, len(self.list_of_units)):
                models_names.append(f"{self.list_of_units[i]},{self.list_of_units[j]}")
                model = Sequential()
                model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))

                model.add(Conv2D(128, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))

                model.add(Flatten())

                model.add(Dense(self.list_of_units[i]))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

                model.add(Dense(self.list_of_units[j]))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.25))

                model.add(Dense(7, activation="softmax"))
                opt = Adam(learning_rate=0.0005)
                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                epochs = 20
                callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
                cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                        validation_data=(X_val, y_val), callbacks=[callback])
                results[i] = cur_history

            plt.figure()
            for result in results:
                plt.plot(result.history["val_accuracy"])
            plt.title("Model Accuracy by Units of Fully Connected Layers")
            plt.xlabel("Epochs")
            plt.ylabel("Validation Accuracy")
            plt.legend(models_names, loc='upper left')
            plt.show()
            for name, result in zip(models_names, results):
                print(
                    f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def dropout(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_dropouts)
        models_names = []
        for i in range(len(self.list_of_dropouts)):
            models_names.append(f"{self.list_of_dropouts[i]} Dropout")
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(self.list_of_dropouts[i]))
            model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(self.list_of_dropouts[i]))

            model.add(Flatten())

            model.add(Dense(64))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.list_of_dropouts[i]))

            model.add(Dense(7, activation="softmax"))

            opt = Adam(learning_rate=0.0005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Size of Dropout")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def max_pooling_pool_size(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_max_pooling)
        models_names = []
        for i in range(len(self.list_of_max_pooling)):
            max_pool = self.list_of_max_pooling[i]

            models_names.append(f"{self.list_of_max_pooling[i]}pool size")
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(max_pool, max_pool), strides=max_pool))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(max_pool, max_pool), strides=max_pool))
            model.add(Dropout(0.25))

            model.add(Flatten())

            model.add(Dense(64))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

            model.add(Dense(7, activation="softmax"))

            opt = Adam(learning_rate=0.0005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Max Pooling Parameters")
        plt.xlabel("epochs")
        plt.ylabel("validation accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def learning_rate(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_lr)
        models_names = []

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(7, activation="softmax"))

        for i, lr in enumerate(self.list_of_lr):
            models_names.append(f"lr = {lr}")

            opt = Adam(learning_rate=lr)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=self.default_batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def batch_size(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.4, random_state=11)
        results = [0] * len(self.list_of_batch_sizes)
        models_names = []

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(7, activation="softmax"))

        for i, batch_size in enumerate(self.list_of_batch_sizes):
            models_names.append(f"batchsize = {batch_size}")

            opt = Adam(learning_rate=0.005)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = 20
            callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(X_val, y_val), callbacks=[callback])
            results[i] = cur_history

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Batch Size")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")


def main_1():
    experiment = ArchitectureExperiments()
    print("Experiment_1: Number of Convolutional Layers")
    experiment.number_of_conv_layers()
    # print("Experiment_2: Number of Convolutional Filters")
    # experiment.filters()
    # print("Experiment_3: Max Pooling - Pool Size and Stride")
    # experiment.max_pooling_pool_size()
    # print("Experiment_4: Number of Fully Connected Layers")
    # experiment.number_of_fully_connected_layers()
    # print("Experiment_5: Number of Fully Connected Units")
    # experiment.dense_units()
    # print("Experiment_6: Dropout")
    # experiment.dropout()
    # print("Experiment_7: Learning Rate")
    # experiment.learning_rate()
    # print("Experiment_8: Batch Size")
    # experiment.batch_size()

    print("Done.")


if __name__ == '__main__':
    main_1()
