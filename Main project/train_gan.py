"""
Code to train the GAN model.

To use: Change the path and run the file. Once it has finished running the two models (discriminator and generator)
will be saved in the models folder.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import sys
sys.append("tools")
from working_simple_gan import get_gan, train_gan
from utils import get_pcam_generators

image_size = (32,32)
batch_size = 128
latent_dim = 500
gan, discriminator, generator = get_gan()
epochs = 300
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject\train+val',
                                         image_size, batch_size, batch_size)
train_gan(discriminator, generator, gan, epochs, batch_size, latent_dim, train_gen)
