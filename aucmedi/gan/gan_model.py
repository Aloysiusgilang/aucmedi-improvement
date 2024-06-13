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
    
    def __init__(self, channels, input_shape, loss, metrics, optimizer, batch_size, architecture='DCGAN', encoding_dims=100, step_channels=64):
        
        self.channels = channels
        self.input_shape = input_shape
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        arch_paras = {
            "channels":channels,
            "encoding_dims":encoding_dims,
            "step_channels":step_channels, 
            "optimizer": optimizer, 
            "metrics":metrics, 
            "loss":loss
        }

        if input_shape is not None : arch_paras["input_shape"] = input_shape

        if isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        else:
            self.architecture = architecture

    def train(self, training_generator, epochs=20):
        self.architecture.train(training_generator, epochs)

    def generate(self, num_images, image_class=None, image_format="jpg"):
        noise = np.random.normal(0, 1, (num_images, 100))

        generated = self.architecture.generator.predict(noise)
        #save the genearted images to output directory
        augmented_images = []
        for i in range(num_images):
            # TODO: handle stik image type
            filename = f"{image_class}_{i}.{image_format}"
            # TODO: handle standardization mode
            cv2.imwrite(os.path.join(self.output_directory, filename), generated[i] * 255)
            augmented_images.append(filename)

        return augmented_images
    
    def save_model(self, model_path):
        self.architecture.generator.save(model_path)
        self.architecture.disciminator.save(model_path)
        self.architecture.combined.save(model_path)

    def load_model(self, model_path="output/model.keras"):
        self.architecture.generator = load_model(model_path)
        self.architecture.disciminator = load_model(model_path)
        self.architecture.combined = load_model(model_path)
    