from tensorflow.keras.layers.experimental import preprocessing
from numpy import expand_dims
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from os import listdir
import re


class AugmentationExperiments:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.img_size = 48
        self.list_of_zoom_ranges = [0.4, 0.5, 0.6]
        self.list_of_shift_ranges = [0.2, 0.5]
        self.list_of_rotate_ranges = [30, 40, 50]
        self.one_hot_dict = {"0": "Angry", "1": "Disgust", "2": "Fear", "3": "Happy", "4": "Neutral", "5": "Sad",
                             "6": "Surprise"}

        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.batch_size,
                                                            class_mode='categorical', shuffle=True)
        total_images = train_generator.n
        steps = total_images // self.batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        self.X = np.array(X)
        self.y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=11)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if not os.path.exists("new_train"):
            os.mkdir("new_train")
            directories = listdir("train/")
            for directory in range(1, len(directories)):
                os.mkdir("new_train/" + f"{directories[directory]}")

        # remove files
        directories = listdir("new_train/")
        for directory in range(len(directories)):
            subdir = listdir("new_train/" + f"{directories[directory]}")
            for file in subdir:
                if re.search("new", file):
                    os.remove("new_train/" + f"{directories[directory]}/" + file)

    def shift_images(self):
        results = [0] * len(self.list_of_shift_ranges)
        models_names = []
        for r in range(len(self.list_of_shift_ranges)):

            for i, img in enumerate(self.X_train):
                label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
                # plt.imshow(img, cmap="gray")
                # plt.show()
                shift_img(img, label, self.list_of_shift_ranges[r], i)
                if i == 10:
                    break

            datagen_train = ImageDataGenerator()
            train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                                color_mode="grayscale", batch_size=self.batch_size,
                                                                class_mode='categorical', shuffle=True)
            total_images = train_generator.n
            steps = total_images // self.batch_size
            X, y = [], []
            for i in range(round(steps)):
                a, b = train_generator.next()
                X.extend(a)
                y.extend(b)
            new_X = np.array(X)
            new_y = np.array(y)

            X_train = np.concatenate((self.X_train, new_X), axis=0)
            y_train = np.concatenate((self.y_train, new_y), axis=0)

            models_names.append(f"{self.list_of_shift_ranges[r]} Shift Range")
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
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                    validation_data=(self.X_val, self.y_val), callbacks=[callback])
            results[r] = cur_history

            # remove files
            directories = listdir("new_train/")
            for directory in range(len(directories)):
                subdir = listdir("new_train/"+f"{directories[directory]}")
                for file in subdir:
                    if re.search("new", file):
                        os.remove("new_train/"+f"{directories[directory]}/"+file)

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Shift Range")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def rotate_images(self):
        results = [0] * len(self.list_of_rotate_ranges)
        models_names = []
        for r in range(len(self.list_of_rotate_ranges)):

            for i, img in enumerate(self.X_train):
                label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
                # plt.imshow(img, cmap="gray")
                # plt.show()
                random_rotate_img(img, label, self.list_of_rotate_ranges[r], i)

            datagen_train = ImageDataGenerator()
            train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                                color_mode="grayscale", batch_size=self.batch_size,
                                                                class_mode='categorical', shuffle=True)
            total_images = train_generator.n
            steps = total_images // self.batch_size
            X, y = [], []
            for i in range(round(steps)):
                a, b = train_generator.next()
                X.extend(a)
                y.extend(b)
            new_X = np.array(X)
            new_y = np.array(y)

            X_train = np.concatenate((self.X_train, new_X), axis=0)
            y_train = np.concatenate((self.y_train, new_y), axis=0)

            models_names.append(f"{self.list_of_rotate_ranges[r]} Rotation Range")
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
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                    validation_data=(self.X_val, self.y_val), callbacks=[callback])
            results[r] = cur_history

            # remove files
            directories = listdir("new_train/")
            for directory in range(len(directories)):
                subdir = listdir("new_train/"+f"{directories[directory]}")
                for file in subdir:
                    if re.search("new", file):
                        os.remove("new_train/"+f"{directories[directory]}/"+file)

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Rotation Range")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def zoom_images(self):
        results = [0] * len(self.list_of_zoom_ranges)
        models_names = []
        for r in range(len(self.list_of_zoom_ranges)):

            for i, img in enumerate(self.X_train):
                label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
                # plt.imshow(img, cmap="gray")
                # plt.show()
                random_zoom_img(img, label, self.list_of_zoom_ranges[r], i)

            datagen_train = ImageDataGenerator()
            train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                                color_mode="grayscale", batch_size=self.batch_size,
                                                                class_mode='categorical', shuffle=True)
            total_images = train_generator.n
            steps = total_images // self.batch_size
            X, y = [], []
            for i in range(round(steps)):
                a, b = train_generator.next()
                X.extend(a)
                y.extend(b)
            new_X = np.array(X)
            new_y = np.array(y)

            X_train = np.concatenate((self.X_train, new_X), axis=0)
            y_train = np.concatenate((self.y_train, new_y), axis=0)

            models_names.append(f"{self.list_of_zoom_ranges[r]} Zoom Range")
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
            cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                    validation_data=(self.X_val, self.y_val), callbacks=[callback])
            results[r] = cur_history

            # remove files
            directories = listdir("new_train/")
            for directory in range(len(directories)):
                subdir = listdir("new_train/"+f"{directories[directory]}")
                for file in subdir:
                    if re.search("new", file):
                        os.remove("new_train/"+f"{directories[directory]}/"+file)

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Zoom Range")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")

    def augmentations_experiment(self, zoom_range, shift_range, rotation_range):
        results = [0] * 4
        models_names = []

        # random zoom model
        for i, img in enumerate(self.X_train):
            label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
            # plt.imshow(img, cmap="gray")
            # plt.show()
            random_zoom_img(img, label, zoom_range, i)

        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.batch_size,
                                                            class_mode='categorical', shuffle=True)
        total_images = train_generator.n
        steps = total_images // self.batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        new_X = np.array(X)
        new_y = np.array(y)

        X_train = np.concatenate((self.X_train, new_X), axis=0)
        y_train = np.concatenate((self.y_train, new_y), axis=0)

        models_names.append("Random Zoom")
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
        cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                    validation_data=(self.X_val, self.y_val), callbacks=[callback])
        results[0] = cur_history

        # remove files
        directories = listdir("new_train/")
        for directory in range(len(directories)):
            subdir = listdir("new_train/" + f"{directories[directory]}")
            for file in subdir:
                if re.search("new", file):
                    os.remove("new_train/" + f"{directories[directory]}/" + file)

        # shift model
        for i, img in enumerate(self.X_train):
            label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
            # plt.imshow(img, cmap="gray")
            # plt.show()
            shift_img(img, label, shift_range, i)

        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.batch_size,
                                                            class_mode='categorical', shuffle=True)
        total_images = train_generator.n
        steps = total_images // self.batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        new_X = np.array(X)
        new_y = np.array(y)

        X_train = np.concatenate((self.X_train, new_X), axis=0)
        y_train = np.concatenate((self.y_train, new_y), axis=0)

        models_names.append("Shifting")
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
        cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                validation_data=(self.X_val, self.y_val), callbacks=[callback])
        results[1] = cur_history

        # remove files
        directories = listdir("new_train/")
        for directory in range(len(directories)):
            subdir = listdir("new_train/" + f"{directories[directory]}")
            for file in subdir:
                if re.search("new", file):
                    os.remove("new_train/" + f"{directories[directory]}/" + file)

        # random rotate model
        for i, img in enumerate(self.X_train):
            label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
            # plt.imshow(img, cmap="gray")
            # plt.show()
            random_rotate_img(img, label, rotation_range, i)

        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.batch_size,
                                                            class_mode='categorical', shuffle=True)
        total_images = train_generator.n
        steps = total_images // self.batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        new_X = np.array(X)
        new_y = np.array(y)

        X_train = np.concatenate((self.X_train, new_X), axis=0)
        y_train = np.concatenate((self.y_train, new_y), axis=0)

        models_names.append("Random Rotate")
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
        cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                validation_data=(self.X_val,self.y_val), callbacks=[callback])
        results[2] = cur_history

        # remove files
        directories = listdir("new_train/")
        for directory in range(len(directories)):
            subdir = listdir("new_train/" + f"{directories[directory]}")
            for file in subdir:
                if re.search("new", file):
                    os.remove("new_train/" + f"{directories[directory]}/" + file)

        # horizontal flip model
        for i, img in enumerate(self.X_train):
            label = self.one_hot_dict[str(np.argmax(self.y_train[i]))]
            # plt.imshow(img, cmap="gray")
            # plt.show()
            flip_img(img, label, i)

        datagen_train = ImageDataGenerator()
        train_generator = datagen_train.flow_from_directory("new_train/", target_size=(self.img_size, self.img_size),
                                                            color_mode="grayscale", batch_size=self.batch_size,
                                                            class_mode='categorical', shuffle=True)
        total_images = train_generator.n
        steps = total_images // self.batch_size
        X, y = [], []
        for i in range(round(steps)):
            a, b = train_generator.next()
            X.extend(a)
            y.extend(b)
        new_X = np.array(X)
        new_y = np.array(y)

        X_train = np.concatenate((self.X_train, new_X), axis=0)
        y_train = np.concatenate((self.y_train, new_y), axis=0)

        models_names.append("Horizontal Flip")
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
        cur_history = model.fit(x=X_train, y=y_train, epochs=epochs,
                                validation_data=(self.X_val, self.y_val), callbacks=[callback])
        results[3] = cur_history

        # remove files
        directories = listdir("new_train/")
        for directory in range(len(directories)):
            subdir = listdir("new_train/" + f"{directories[directory]}")
            for file in subdir:
                if re.search("new", file):
                    os.remove("new_train/" + f"{directories[directory]}/" + file)

        plt.figure()
        for result in results:
            plt.plot(result.history["val_accuracy"])
        plt.title("Model Accuracy by Augmentation")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(models_names, loc='upper left')
        plt.show()
        for name, result in zip(models_names, results):
            print(
                    f"{name}: Train Accuracy = {max(result.history['accuracy'])} , Validation Accuracy = {max(result.history['val_accuracy'])}")


'''
autgmentation - one image
'''


def shift_img(image, label, shift_range, index):
    # expand dimension to one sample
    samples = expand_dims(image, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=shift_range, height_shift_range=shift_range, fill_mode='nearest')
    for batch in datagen.flow(samples, batch_size=1, save_to_dir="new_train/"+f"{label}", save_prefix="new"+f"{index}",
                              save_format="jpg"):
        img = batch[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        return img


def random_rotate_img(image, label, random_range, index):
    # expand dimension to one sample
    samples = expand_dims(image, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(rotation_range=random_range, fill_mode='nearest')
    for batch in datagen.flow(samples, batch_size=1, save_to_dir="new_train/"+f"{label}", save_prefix="new"+f"{index}",
                              save_format="jpg"):
        img = batch[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        return img


def flip_img(image, label, index):
    # expand dimension to one sample
    samples = expand_dims(image, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(horizontal_flip=True)
    for batch in datagen.flow(samples, batch_size=1, save_to_dir="new_train/"+f"{label}", save_prefix="new"+f"{index}",
                              save_format="jpg"):
        img = batch[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        return img


def random_zoom_img(image, label, random_range, index):
    # expand dimension to one sample
    samples = expand_dims(image, 0)
    # create image data augmentation generator
    range_list = [random_range, 1]
    datagen = ImageDataGenerator(zoom_range=range_list)
    for batch in datagen.flow(samples, batch_size=1, save_to_dir="new_train/"+f"{label}", save_prefix="new"+f"{index}",
                              save_format="jpg"):
        img = batch[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        return img


def main():
    experiment = AugmentationExperiments(64)
    print("Experiment_1: Shift range")
    experiment.shift_images()
    # print("Experiment_2: Zoom range")
    # experiment.zoom_images()
    # print("Experiment_3: Rotation range")
    # experiment.rotate_images()
    # print("Experiment_4: Different augmentations")
    # experiment.augmentations_experiment(0.5, 0.2, 30)
    # print("Done.")


if __name__ == '__main__':
    main()
