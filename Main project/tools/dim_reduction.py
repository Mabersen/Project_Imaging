"""
Definition of the necessary functions to perform weight evaluation and visualisation of a given model.
    
@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import keras
from keras.layers import Conv2D
from keras import initializers
import tensorflow as tf
from math import ceil
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import sklearn.decomposition as decomp

from custom_layers import MinibatchDiscrimination
from utils import plotImages, plot_images_ordered

def get_order(weights):
    """
    Performs PCA on the input weights and returns an ordering based on the first principal component.
    Args:
        weights:

    Returns:
        pca_ordering: the determined order as an array

    """
    pca = decomp.PCA(n_components=1)
    pca_ordering = np.argsort(pca.fit_transform(
        np.abs(weights)[0, 0])[:, 0])
    return pca_ordering


def get_weights(model, layer):
    """
    Returns the weights for the specified and the directly following convolutional layers of the
    network (skips over any other layers).
    Args:
        model: Model to extract weights from
        layer: Desired convolutional layer to extract weights from

    Returns:
        layer_weights: weights of the desired convolutional layer
        next_layer_weights: weights of the convolutional layer directly after the desired one. Returns False if the last
        layer is selected

    """
    # model.layers.weights[0] is the slopes and model.layers.weights[1] is the biases.
    # make the weights positve and normalize to a [0, 1] range
    conv_indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, Conv2D)]
    first_layer = conv_indices[layer]
    layer_weights = model.layers[first_layer].weights[0] + model.layers[first_layer].weights[1]
    if conv_indices[-1] == first_layer:
        next_layer_weights = False
    else:
        next_layer = conv_indices[layer + 1]
        next_layer_weights = disc.layers[next_layer].weights[0] + disc.layers[next_layer].weights[1]
    return layer_weights, next_layer_weights


def preprocess_images(images):
    """
    Preprocesses images so they can be properly displayed
    Args:
        images: images to be processed

    Returns:
        images: the preprocessed images

    """
    images = images * 0.5 + 0.5
    images = (images - np.amin(images)) / (np.amax(images) - np.amin(images))
    return images


def visualize_first_layer_weights(model):
    """
    Visualizes the weights of the first convolutional layer of a model, ordered by the first PCA component of the
    second convolutional layer.
    Args:
        model: model that the weights should be taken from
    Returns: None

    """
    first_layer_weights, second_layer_weights = get_weights(model, 0)
    images = preprocess_images(first_layer_weights)
    order = get_order(second_layer_weights)
    plot_images_ordered(images, order)


def visualize_layer_weights(layer, n_images, model):
    """
    Visualizes layer weights for any arbitrary layer by performing dimensionality reduction on them beforehand.
    Dimensionality reduction is done through one-sided NMF.
    Args:
        layer: desired layer to be visualized
        n_images: number of desired kernels to be visualized
        model: model that the weights should be taken from.

    Returns:

    """
    nmf = decomp.NMF(3, init='nndsvd')
    w_2, _ = get_weights(model, layer)
    # concatenation hack to make the matrices positive so one sided nmf can be used
    w_2 = np.concatenate([np.maximum(0, w_2), np.maximum(0, -w_2)], axis=2)
    print(w_2.shape)
    w_2 = np.reshape(w_2, (9, w_2.shape[2], w_2.shape[3]))
    images = np.zeros((n_images, 3, 3, 3))
    root = ceil(np.sqrt(n_images))
    dim = [root, root]
    figsize = (root, root)
    for i in range(n_images):
        w_nmf = nmf.fit_transform(w_2[..., i])
        image = np.reshape(w_nmf, (3, 3, 3))
        images[i] = image
    images = preprocess_images(images)
    plotImages(images, dim=dim, figsize=figsize)


# load the models
init = initializers.get("glorot_uniform")
disc = keras.models.load_model('logs/old models/gan_discriminator_epoch_lower_lr145.h5',
                               custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                               'GlorotUniform': init})
classifier = keras.models.load_model('regular_classifer.h5',
                                     custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                     'GlorotUniform': init})

visualize_first_layer_weights(disc)
visualize_first_layer_weights(classifier)
visualize_layer_weights(0, 64, disc)
visualize_layer_weights(0, 64, classifier)
visualize_layer_weights(3, 64, disc)
visualize_layer_weights(3, 64, classifier)
