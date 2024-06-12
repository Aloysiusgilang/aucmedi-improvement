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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
# Internal libraries/scripts
from aucmedi.neural_network.gan_architectures import architecture_dict

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network
class GANNeuralNetwork:
    
    def __init__(self, channels, input_shape=None, architecture='DCGAN', loss="categorical_crossentropy", encoding_dims=100, step_channels=64,metrics=["categorical_accuracy"], learning_rate=1e-4, batch_size=32, output_directory=None):
        
        # Cache parameters
        self.channels = channels
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_directory = output_directory
    
        
        # Assemble architecture parameters
        arch_paras = {"channels":channels, "encoding_dims":encoding_dims, "step_channels":step_channels}
        if input_shape is not None : arch_paras["input_shape"] = input_shape

        if isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        # Initialize passed architecture as parameter
        else:
            self.architecture = architecture

        self.generator, self.discriminator, self.combined = self.architecture.build_gan()

    def train(self, training_generator, epochs=20):
        # Train the GAN

        for epoch in range(epochs):
            for i in range(len(training_generator)):

                # retrieve batches of real imgs
                real_imgs, labels = training_generator[i]
                current_batch_size = len(real_imgs)

                # generate noise for the generator
                noise = np.random.normal(0, 1, (current_batch_size, 100))

                # generate fake imgs
                gen_imgs = self.generator.predict(noise)

                # train the discriminator on real imgs 
                real_y = np.ones((current_batch_size, 1))
                d_loss_real = self.discriminator.train_on_batch(real_imgs, real_y)

                # train the discriminator on fake imgs
                fake_y = np.zeros((current_batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_y)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train the generator
                g_loss = self.combined.train_on_batch(noise, np.ones((current_batch_size, 1)))

                # print progress
                print(f"Epoch {epoch}, Batch {i}, D_Loss: {d_loss[0]}, G_Loss: {g_loss}")


    #---------------------------------------------#
    #               Model Management              #
    #---------------------------------------------#
    # Re-initialize model weights
    def reset_weights(self):
        """ Re-initialize weights of the neural network model.

        Useful for training multiple models with the same NeuralNetwork object.
        """
        self.model.set_weights(self.initialization_weights)

    # Dump model to file
    def dump(self, file_path):
        """ Store model to disk.

        Recommended to utilize the file format ".hdf5".

        Args:
            file_path (str):    Path to store the model on disk.
        """
        self.model.save(file_path)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        """ Load neural network model and its weights from a file.

        After loading, the model will be compiled.

        If loading a model in ".hdf5" format, it is not necessary to define any custom_objects.

        Args:
            file_path (str):            Input path, from which the model will be loaded.
            custom_objects (dict):      Dictionary of custom objects for compiling
                                        (e.g. non-TensorFlow based loss functions or architectures).
        """
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)

    def generate(self, num_images, image_class=None, image_format="jpg"):
        noise = np.random.normal(0, 1, (num_images, 100))

        generated = self.generator.predict(noise)
        #save the genearted images to output directory
        augmented_images = []
        for i in range(num_images):
            # TODO: handle stik image type
            filename = f"{image_class}_{i}.{image_format}"
            cv2.imwrite(os.path.join(self.output_directory, filename), generated[i] * 255)
            augmented_images.append(filename)

        return augmented_images
    