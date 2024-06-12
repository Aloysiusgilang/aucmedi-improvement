#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
# Internal libraries/scripts
from aucmedi.gan.gan_architectures import architecture_dict

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network
class GANNeuralNetwork:
    
    def __init__(self, channels, input_shape, loss, metrics, optimizer, batch_size, output_directory, architecture='DCGAN', encoding_dims=100, step_channels=64):
        
        # Cache parameters
        self.channels = channels
        self.input_shape = input_shape
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.output_directory = output_directory
    
        
        # Assemble architecture parameters
        arch_paras = {"channels":channels, "encoding_dims":encoding_dims, "step_channels":step_channels, "optimizer": optimizer, "metrics":metrics, "loss":loss}
        if input_shape is not None : arch_paras["input_shape"] = input_shape

        if isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        # Initialize passed architecture as parameter
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
            cv2.imwrite(os.path.join(self.output_directory, filename), generated[i] * 255)
            augmented_images.append(filename)

        return augmented_images
    