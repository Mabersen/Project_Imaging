import gzip
import os
import pickle
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from keras import layers
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 96
batch_size = 500


def get_pcam_generators(base_dir, train_batch_size=batch_size, val_batch_size=batch_size):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=train_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


def loadPatchCamelyon(path):
    f = gzip.open(path, 'rb')
    train_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject')


def crop(gen, percentage):
    for x in gen:
        yield tf.image.central_crop(x[0], percentage), x[1]


crop_train_gen = crop(train_gen, 0.333)
crop_val_gen = crop(val_gen, 0.333)


def saveModels(epoch):
    generator.save('gan_generator_epoch_{}.h5'.format(epoch))
    discriminator.save('gan_discriminator_epoch_{}.h5'.format(epoch))


def plotImagesPatchCamelyon(images, dim=(10, 10), figsize=(10, 10), title=''):
    images = images.astype(np.float32) * 0.5 + 0.5
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def plotGeneratedImagesPatchCamelyon(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 32, 32, 3)
    generatedImages = generatedImages.astype(np.float32) * 0.5 + 0.5

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle('Epoch {}'.format(epoch))
    plt.show()


def get_discriminator_histopathology(
        in_shape=(32, 32, 3)):
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
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_generator_histopathology():
    inputs = Input(shape=(latent_dim,))
    x = Dense(128 * 4 * 4, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    outputs = Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh')(x)  # 3 filters for 3 channels (rgb)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


latent_dim = 400

discriminator = get_discriminator_histopathology()
generator = get_generator_histopathology()

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8))
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8))

discriminator.trainable = False
z = keras.layers.Input(shape=(latent_dim,))
x = generator(z)
D_G_z = discriminator(x)
gan = keras.models.Model(inputs=z, outputs=D_G_z)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8))

# d_losses = []
# g_losses = []
#
# epochs = 200
# batch_size = 128
#
# X_train = (train_set_images.reshape(train_set_images.shape[0], train_set_images.shape[1], train_set_images.shape[2],
#                                     train_set_images.shape[3]).astype(np.float32) - 0.5) / 0.5
# # map pixel values to the [-1, 1] range to be compatible with tanh activation function
#
# batch_count = int(X_train.shape[0] / batch_size)
# print(batch_count)
# for e in range(epochs):
#     for b in range(batch_count):
#         # Get a random set of input noise and images
#         noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
#         image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
#
#         # Generate some fake histopathology images using the generator
#         generated_images = generator.predict(noise)
#
#         # Concatenate the fake and real images
#         X = np.concatenate([image_batch, generated_images])
#
#         # Labels for generated and real data
#         y_dis = np.zeros(2 * batch_size)
#         # Set reference to 1 for real samples
#         y_dis[:batch_size] = 1
#
#         # Train discriminator with this batch of samples
#         discriminator.trainable = True
#         d_loss = discriminator.train_on_batch(X, y_dis)
#
#         # Train generator with a new batch of generated samples
#         noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
#
#         # From the generator's perspective, the discriminator should predict
#         # ones for all samples
#         y_gen = np.ones(batch_size)
#
#         # Freeze the discriminator part
#         discriminator.trainable = False
#
#         # Train the GAN to predict ones
#         g_loss = gan.train_on_batch(noise, y_gen)
#
#         # Store loss of most recent batch from this epoch
#         if int(b * 100 / batch_count) % 5 == 0:
#             print(str(int(b * 100 / batch_count)) + '% done with epoch number' + str(e))
#     d_losses.append(d_loss)
#     g_losses.append(g_loss)
#
#     if e % 5 == 0:
#         noise = np.random.normal(0, 1, size=[100, latent_dim])
#         generatedImages = generator.predict(noise)
#         generatedImages = generatedImages.reshape(100, 28, 28, 3)
#         plotImagesPatchCamelyon(generatedImages,
#                                 title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
#         display.display(plt.gcf())
#         display.clear_output(wait=True)
#         time.sleep(0.001)
#         saveModels(e)
#
# plotGeneratedImagesPatchCamelyon(epochs)
d_losses = []
g_losses = []

epochs = 5

# map pixel values to the [-1, 1] range to be compatible with tanh activation function

batch_count = 144000 // (batch_size*10)
for e in range(epochs):
    for b in range(batch_count):
        start_time = time.time()
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        image_batch = tf.image.central_crop(next(train_gen)[0], 0.333)

        # Generate some fake histopathology images using the generator
        generated_images = generator.predict(noise)

        # Concatenate the fake and real images
        X = np.concatenate([image_batch, generated_images])

        # Labels for generated and real data
        y_dis = np.zeros(2 * batch_size)
        # Set reference to 1 for real samples
        y_dis[:batch_size] = 1

        # Train discriminator with this batch of samples
        discriminator.trainable = True
        try:
            d_loss = discriminator.train_on_batch(X, y_dis)
        except ValueError:
            print(image_batch.shape, noise.shape)
            continue
        # Train generator with a new batch of generated samples
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

        # From the generator's perspective, the discriminator should predict
        # ones for all samples
        y_gen = np.ones(batch_size)

        # Freeze the discriminator part
        discriminator.trainable = False

        # Train the GAN to predict ones
        g_loss = gan.train_on_batch(noise, y_gen)
        end_time = time.time()
        time_1_batch = end_time - start_time
        # Store loss of most recent batch from this epoch
        if int(b * 100 / batch_count) % 5 == 0:
            print(str(int(b * 100 / batch_count)) + '% done with epoch number' + str(e))
            print('eta: ' + str(
                (epochs - (e + 1)) * time_1_batch * batch_count + (time_1_batch * batch_count - b)) + ' seconds')
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if e % 5 == 0:
        noise = np.random.normal(0, 1, size=[100, latent_dim])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(100, 32, 32, 3)
        plotImagesPatchCamelyon(generatedImages,
                                title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.001)
        saveModels(e)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(d_losses)
plt.title('Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(g_losses)
plt.title('Generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print(d_losses, g_losses)
plotGeneratedImagesPatchCamelyon(epochs)


def add_noise_2(model):
    for layer in model.trainable_weights:
        noise = np.random.normal(loc=0.0, scale=0.02, size=layer.shape)
        layer.assign_add(noise)


def transfer_classifier(transfer_source):
    transfer_source.trainable = False
    inputs = transfer_source.inputs
    # take the output from the second to last layer of the full model and
    # replace the last layer with a sigmoid 1-unit dense layer for classification of the PCAM set
    outputs = Dense(1, activation='sigmoid')(transfer_source.layers[-2].output)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    #add_noise_2(model)
    return model


def classifier():
    model = get_discriminator_histopathology()
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    return model


steps = 144000 // batch_size
transfer = transfer_classifier(discriminator)
standard = classifier()
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
# do initial training of the transfer model
transfer.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_1])
# unfreeze the layers of the transfer model and finetune
for layer in transfer.layers:
    layer.trainable = True
transfer.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
transfer.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_1])
standard.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_2])
standard.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
standard.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_2])
transfer.summary()
standard.summary()
transfer.evaluate(crop_val_gen, steps=steps, batch_size=128)
standard.evaluate(crop_val_gen, steps=steps, batch_size=128)