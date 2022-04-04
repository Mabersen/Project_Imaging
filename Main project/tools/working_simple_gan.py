"""
Code to create the GAN. Defines the functions to obtain the discriminator, the generator, the GAN (combined model) 
and to train the GAN.

To use: Change the path and run the file. Once it has finished running the two models (discriminator and generator)
will be saved in the models folder.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import time
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, Reshape, BatchNormalization
from keras import Model

from custom_layers import MinibatchDiscrimination, UpSampleBlock
from utils import saveModels, plotGeneratedImagesPatchCamelyon, plotImages

def get_discriminator_histopathology(
        in_shape=(32, 32, 3,)):
    """
    Constructs and returns the discriminator network used in the GAN.
    Args:
        in_shape: shape of the input tensor, channels-last.

    Returns:
        discriminator: the discriminator network, not yet compiled.
    """
    inputs = Input(shape=in_shape)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape,
               kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    # input shape accounts for the three channels
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = MinibatchDiscrimination(5, 3)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_generator_histopathology(latent_dim):
    """
    Constructs and returns the generator network to be used in the GAN.
    Returns:
        generator: the generator network, not yet compiled.

    """
    inputs = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 128, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 128))(x)
    x = UpSampleBlock(128)(x)
    x = UpSampleBlock(64)(x)
    x = UpSampleBlock(64)(x)
    outputs = Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh')(x)  # 3 filters for 3 channels (rgb)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


def get_gan(input_size=(32, 32, 3,), latent_dim=500, disc_lr=0.0004, gen_lr=0.0001):
    """

    Args:
        input_size: size of the discriminator input, channels last.
        latent_dim: dimensionality of the input space
        disc_lr: learning rate for the Adam optimizer for the discriminator
        gen_lr: learning rate for the Adam optimizer for the generator

    Returns:
        gan: the compiled combined gan model
        discriminator: the compiled discriminator model
        generator: the compiled generator model

    """
    # Get the component networks
    discriminator = get_discriminator_histopathology(input_size)
    generator = get_generator_histopathology(latent_dim)
    # Compile the component networks
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5, beta_2=0.999))
    generator.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5, beta_2=0.999))
    # Make sure the discriminator does not update when training the generator.
    discriminator.trainable = False
    # create input layer
    z = keras.layers.Input(shape=(latent_dim,))
    x = generator(z)
    D_G_z = discriminator(x)
    gan = keras.models.Model(inputs=z, outputs=D_G_z)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5,
                                                                               beta_2=0.999))
    return gan, discriminator, generator


def train_gan(discriminator, generator, gan, epochs, batch_size, latent_dim, train_gen):
    """
    Performs the training of the GAN model.
    Args:
        discriminator: discriminator network for the GAN
        generator: generator network for the GAN
        gan: combined network for the GAN
        epochs: Number of epochs to perform training for
        batch_count: Number of batches per epoch
        latent_dim: dimensionality of the input space for the generator
        train_gen: generator for the training dataset.

    Returns:

    """
    d_losses_real = []
    d_losses_fake = []
    g_losses = []
    batch_times = []
    batch_count = 144000 // batch_size
    for e in range(epochs):
        for b in range(batch_count):
            start_time = time.time()
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            image_batch = next(train_gen)[0]

            # Generate some fake histopathology images using the generator
            generated_images = generator.predict(noise)

            # Concatenate the fake and real images
            X_real = image_batch
            X_fake = generated_images

            # Labels for generated and real data
            y_real = np.zeros(batch_size)
            # smooth labels
            y_real[:] = 0.9
            y_fake = np.zeros(batch_size)

            # Train discriminator with this batch of samples
            # training has to be seperated between real and fake images for the minibatch discrimination to work
            discriminator.trainable = True
            try:
                d_loss_real = discriminator.train_on_batch(X_real, y_real)
                d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            except ValueError:
                print(X_real.shape, X_fake.shape, noise.shape)
                continue
            # Train generator with a new batch of generated samples
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

            # From the generator's perspective, the discriminator should predict
            # ones for all samples
            y_gen = np.ones(batch_size)

            # Freeze the discriminator part
            discriminator.trainable = False

            # train the discriminator exclusively for the first few batches
            if e == 0 and b < 20:
                end_time = time.time()
                batch_times.append(end_time - start_time)
                continue
            # Train the GAN to predict ones
            g_loss = gan.train_on_batch(noise, y_gen)
            end_time = time.time()
            batch_times.append(end_time - start_time)
            # Store loss of most recent batch from this epoch
            if int(b * 100 / batch_count) % 2 == 0:
                if b > 2:
                    time_1_batch = np.mean(batch_times[2:])
                else:
                    time_1_batch = batch_times[0]
                n = int(b * 100 / batch_count)
                n_batches_left = ((epochs - (e + 1)) * batch_count) + (batch_count - (b + 1))
                eta = time_1_batch * n_batches_left
                print(
                    '\r%d%% done with epoch number %d, eta: %d seconds, batch time: %f' % (n, e + 1, eta, time_1_batch),
                    end="")
        d_losses_real.append(d_loss_real)
        d_losses_fake.append(d_loss_fake)
        g_losses.append(g_loss)

        if e % 5 == 0:
            noise = np.random.normal(0, 1, size=[100, latent_dim])
            generatedImages = generator.predict(noise)
            generatedImages = generatedImages.reshape(100, 32, 32, 3)
            plotImages(generatedImages,
                       title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.01)
            saveModels(discriminator, generator, e, 'Upsampling')
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(d_losses_fake)
    plt.title('Discriminator loss on fake images')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(g_losses)
    plt.title('Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plotGeneratedImagesPatchCamelyon(epochs, generator)
    return d_losses_real, d_losses_real, g_losses
