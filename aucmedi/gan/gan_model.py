# External libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
# Internal libraries/scripts
from aucmedi.gan.gan_architectures import architecture_dict

#-----------------------------------------------------#
#           GAN Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the GAN Neural Network
class GANNeuralNetwork:
    """ Neural Network class providing functionality for handling all model  methods."""
    
    def __init__(self, input_shape, batch_size, architecture='DCGAN', encoding_dims=128):
        
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.encoding_dims = encoding_dims
        
        arch_paras = {
            "encoding_dims":encoding_dims,
        }

        if input_shape is not None : arch_paras["input_shape"] = input_shape

        if isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        else:
            self.architecture = architecture

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
    