# External libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
# Internal libraries/scripts
from aucmedi.gan.gan_architectures import architecture_dict

#-----------------------------------------------------#
#           GAN Augmentation (model) class            #
#-----------------------------------------------------#
# Class which represents the GAN Augmentation
class GANNeuralNetwork:
    
    def __init__(self, channels, input_shape=None, architecture='DCGAN',
                  encoding_dims=128, step_channels=64, d_optimizer=Adam(), g_optimizer=Adam(), d_loss_fn=tf.keras.losses.BinaryCrossentropy(), 
                  g_loss_fn=tf.keras.losses.BinaryCrossentropy(), 
                  d_loss_metric=tf.keras.metrics.Mean(name="d_loss"), 
                  g_loss_metric=tf.keras.metrics.Mean(name="g_loss")):
        
        self.input_shape = input_shape
        self.channels = channels
        self.encoding_dims = encoding_dims
        
        arch_paras = {
            "encoding_dims":encoding_dims,
            "channels":channels,
            "step_channels":step_channels
        }

        if input_shape is not None : arch_paras["input_shape"] = input_shape

        if isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        else:
            self.architecture = architecture

        self.architecture.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_loss_fn=d_loss_fn, g_loss_fn=g_loss_fn, d_loss_metric=d_loss_metric, g_loss_metric=g_loss_metric)

    def train(self, training_generator, epochs=20):
        self.architecture.fit(training_generator, epochs=epochs)

    def generate(self, noise):
        return self.architecture.generator.predict(noise)
    
    def save_model(self, model_path):
        self.architecture.generator.save(model_path)
        self.architecture.disciminator.save(model_path)
        self.architecture.combined.save(model_path)

    def load_model(self, model_path="output/model.keras"):
        self.architecture.generator = load_model(model_path)
        self.architecture.disciminator = load_model(model_path)
        self.architecture.combined = load_model(model_path)
    