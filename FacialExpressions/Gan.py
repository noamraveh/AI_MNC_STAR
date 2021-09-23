import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Conv2DTranspose, Reshape
import matplotlib.pyplot as plt
from keras.models import load_model


def save_plot(num_examples, epoch_num, n=3):
    examples = (num_examples + 1) / 2.0  # scale from [-1,1] to [0,1]
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i][:, :, 0], cmap='gray')
    filename = 'epoch_%03d_images.png' % (epoch_num + 1)
    plt.savefig(filename)
    plt.close()


class Samples:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def generate_real_images(self, dataset, num_samples):
        """
        chooses randomly num_samples images from the dataset
        :param dataset: the dataset
        :param num_samples: number of samples taken from the dataset
        :return: the randomly chosen <num_samples> images and their label
        """
        indexes = np.random.randint(0, dataset.shape[0], num_samples)
        samples_x = dataset[indexes]
        samples_y = np.ones((num_samples, 1))
        return samples_x, samples_y

    def generate_latent_points(self, num_samples):
        """
        generates latent points as input for the encoder
        :param num_samples: number of samples to generate
        :return: the generated samples
        """
        vectors = np.random.randn(self.latent_dim * num_samples)
        # reshape into a batch of inputs for the network
        vectors = vectors.reshape(num_samples, self.latent_dim)
        return vectors

    def generate_artificial_images(self, encoder, num_samples):
        """
        takes arrays of latent points and creates artificial images of them
        :param encoder: the model which takes the latent points and creates artificial images of them
        :param num_samples: number of artificial images to generate
        :return: the generated artificial images
        """
        x = self.generate_latent_points(num_samples)
        samples_x = encoder.predict(x)
        samples_y = np.zeros((num_samples, 1))
        return samples_x, samples_y


class Gan:
    def __init__(self, dataset, latent_dim):
        self.dataset = dataset
        self.input_shape = (48, 48, 1)
        self.latent_dim = latent_dim
        self.samples = Samples(latent_dim)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.gan_model = self.create_gan()

    def create_encoder(self):
        """
        creates the encoder without compiling it
        :return: encoder
        """
        up_sampling_layers = [Dense(6 * 6 * 256, input_dim=self.latent_dim),
                              LeakyReLU(alpha=0.2),
                              Reshape((6, 6, 256)),
                              Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
                              LeakyReLU(alpha=0.2),
                              Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
                              LeakyReLU(alpha=0.2),
                              Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
                              LeakyReLU(alpha=0.2),
                              Conv2D(1, (3, 3), activation='tanh', padding='same')]
        encoder = Sequential(up_sampling_layers)
        return encoder

    def create_decoder(self):
        """
        creates the decoder
        :return: decoder
        """
        down_sampling_layers = [Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape),
                                LeakyReLU(alpha=0.2),
                                Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
                                LeakyReLU(alpha=0.2),
                                Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
                                LeakyReLU(alpha=0.2),
                                Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
                                LeakyReLU(alpha=0.2),
                                Flatten(),
                                Dropout(0.2),
                                Dense(1, activation='sigmoid')]
        decoder = Sequential(down_sampling_layers)
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        decoder.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return decoder

    def create_gan(self):
        """
        creates the encoder-decoder model
        :return: the generated GAN model
        """
        self.decoder.trainable = False
        gan = Sequential()
        gan.add(self.encoder)
        gan.add(self.decoder)
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan

    def train_gan(self, num_epochs=200, batch_size=20):
        """
        trains the GAN model each time on a batches
        :param num_epochs: number of epochs to perform
        :param batch_size: batch size of images
        """
        batches_per_epoch = int(self.dataset.shape[0] / batch_size)
        half_batch = int(batch_size / 2)
        for epoch_num in range(num_epochs):
            for batch in range(batches_per_epoch):
                X_real, y_real = self.samples.generate_real_images(self.dataset, half_batch)
                self.decoder.train_on_batch(X_real, y_real)
                X_artificial, y_artificial = self.samples.generate_artificial_images(self.encoder, half_batch)
                self.decoder.train_on_batch(X_artificial, y_artificial)
                X_gan, y_gan = self.samples.generate_latent_points(batch_size), np.ones((batch_size, 1))
                self.gan_model.train_on_batch(X_gan, y_gan)
                print(f"{epoch_num+1} : {batch}/{batches_per_epoch}")
            if (epoch_num + 1) % 10 == 0:
                self.analysis(epoch_num)

    def analysis(self, epoch_num, num_samples=50):
        """
        summarizes the training each 10 epochs and print the gained accuracy on real and artificial images.
        saves a weights file for the trained encoder
        :param epoch_num: number of epoch to be used in save_plot function
        :param num_samples: number of samples to evaluate on
        """
        X_real, y_real = self.samples.generate_real_images(self.dataset, num_samples)
        _, acc_real = self.decoder.evaluate(X_real, y_real, verbose=0)
        X_artificial, y_artificial = self.samples.generate_artificial_images(self.encoder, num_samples)
        _, acc_artificial = self.decoder.evaluate(X_artificial, y_artificial, verbose=0)
        print('Accuracy: on real: %.0f%%, on artificial: %.0f%%' % (acc_real * 100, acc_artificial * 100))
        save_plot(X_artificial, epoch_num)
        filename = 'generator_model_%03d.h5' % (epoch_num + 1)
        self.encoder.save(filename)

    def create_images(self, h5_file, n):
        """
        loads the h5 file to a model and creates n*n artificial images.
        it prints a plot of those images and saves them to a directory
        :param h5_file:
        :param n:
        :return:
        """
        model = load_model(h5_file)
        latent_points = self.samples.generate_latent_points(n * n)
        artificial_images = model.predict(latent_points)
        artificial_images = (artificial_images + 1) / 2.0  # scale from [-1,1] to [0,1]
        for i in range(n * n):
            plt.subplot(n, n, 1 + i)
            plt.axis('off')
            plt.imshow(artificial_images[i][:, :, 0], cmap='gray')
            plt.imsave(f'DIS/{i}.png', artificial_images[i][:, :, 0], cmap='gray')
        plt.show()


def main_3():
    img_size = 48
    batch_size = 64
    datagen_train = ImageDataGenerator()
    train_generator = datagen_train.flow_from_directory("train_gan_d/", target_size=(img_size, img_size),
                                                        color_mode="grayscale", batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    total_images = train_generator.n
    steps = total_images / batch_size
    images = []
    for i in range(round(steps)):
        a, _ = train_generator.next()
        images.extend(a)
    images = np.array(images)

    images = images.astype('float32')
    images = (images - 127.5) / 127.5      # scale from [0,255] to [-1,1]

    latent_dim = 100
    gan = Gan(images, latent_dim)
    gan.create_gan()
    gan.train_gan()


    #generating new images based on trained model weights
    #gan.create_images("generator_model_250.h5", 7)


if __name__ == '__main__':
    main_3()
