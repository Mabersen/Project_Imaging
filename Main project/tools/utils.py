# -*- coding: utf-8 -*-
"""
Compilation of useful functions for the 8P361 course. This is divided in:
    - Handling inputs:
        get_pcam_generators
        generate_latent_points
    - Saving the relevant data:
        saveModels
    - Visualising data:
        plotImages
        plot_images_ordered
        plotGeneratedImagesPatchCamelyon

@author: Ari
"""

import os
import keras
import numpy as np
import matplotlib as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def generate_latent_points(latent_dim, n_samples):
    """
    Generate points in latent space as input for a generator

    Parameters
    ----------
    latent_dim : latent dimension, int
    n_samples : number of samples to generate, int

    Returns
    -------
    z_input : batch of inputs, vector

    """
    # generate points in the latent space
    z_input = np.random.normal(size=latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input


def get_pcam_generators(base_dir, image_size,  train_batch_size=128, val_batch_size=128):
    """
    Creates keras ImageDataGenerator objects of the datasets. Assigns labels to images based on the folder they're
    contained in.
    Args:
        base_dir: path to the directory containing the dataset. Dataset directories should have the following structure:
         'base_dir\train+val\' followed by either 'train' or 'val' (training and validation set respectively) and either
         '0' or '1' (class labels) folders. Dataset images should be appropriately  placed into their respective class
         label folders
        train_batch_size: batch size to be used for the training set
        val_batch_size: batch size to be used for the validation set

    Returns:
        train_gen: generator for the training set
        val_gen: generator for the validation set

    """
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')

    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=image_size,
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=image_size,
                                          batch_size=val_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen

def saveModels(discriminator, generator, epoch, name):
    """
    Saves the generator and discriminator models to a keras compatible .h5 file.
    Adapted from: assignments for this course (need better sourcing)

    Args:
        epoch: The training epoch the model is currently on
        name: Name of the model

    Returns: None

    """
    generator.trainable = True
    discriminator.trainable = True
    keras.models.save_model(generator, 'gan_generator_epoch_{name}_{epoch}.h5'.format(name=name, epoch=epoch))
    keras.models.save_model(discriminator, 'gan_discriminator_epoch_{name}_{epoch}.h5'.format(name=name, epoch=epoch))
    generator.trainable = False
    discriminator.trainable = False

def plotImages(images, dim=(10, 10), figsize=(10, 10), title=''):
    """
    Plot a number of images in a grid.
    Source: assignments for this course (need better sourcing)
    Args:
        images: images to be plotted, channels-last.
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches
        title: title to be displayed above the image grid.

    Returns: None

    """
    images = images.astype(np.float32) * 0.5 + 0.5
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    

def plot_images_ordered(images, order, dim=[8, 8], figsize=[8, 8]):
    """
    plots images in a given order.
    Args:
        images: images to be plotted
        order: the order to plot the images in, array.
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches

    Returns:

    """
    plt.figure(figsize=figsize)
    for i, num in enumerate(order):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[:, :, :, num])
        plt.title(num)
        plt.axis('off')
    plt.show()

def plotGeneratedImagesPatchCamelyon(epoch, generator, latent_dim=500, examples=100, dim=(10, 10), figsize=(10, 10)):
    """
    Generates and plots images from a GAN generator network in a grid.
    Args:
        generator: generator network to be used for generating the images
        latent_dim: dimensionality of the input space.
        epoch: training epoch that the images originate from
        examples: number of images to be generated
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches


    Returns: None

    """
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