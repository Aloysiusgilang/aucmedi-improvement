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
#External libraries
import unittest
import tempfile
import os
from PIL import Image
import numpy as np
#Internal libraries
from aucmedi import *
from aucmedi.gan.gan_model import GANNeuralNetwork
import tensorflow as tf


#-----------------------------------------------------#
#              Unittest: NeuralNetwork               #
#-----------------------------------------------------#
class GANNeuralNetworkTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 10):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((10, 4), dtype=np.uint8)
        for i in range(0, 10):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create RGB Data Generator
        self.datagen = DataGenerator(self.sampleList_rgb,
                                     self.tmp_data.name,
                                     labels=self.labels_ohe,
                                     resize=(32, 32),
                                     shuffle=True,
                                     grayscale=False, batch_size=3)

    # #-------------------------------------------------#
    # #                  Model Training                 #
    # #-------------------------------------------------#
    def test_training_dcgan(self):
        model = GANNeuralNetwork(channels=3, input_shape=(32,32), encoding_dims=100, architecture='2D.DCGAN')
        hist = model.train(training_generator=self.datagen,
                           epochs=3)
        self.assertTrue("g_loss" in hist)
        self.assertTrue("d_loss" in hist)

    def test_training_wgan_gp(self):
        # def generator_loss(fake_img):
        #     return -tf.reduce_mean(fake_img)
        model = GANNeuralNetwork(channels=3, input_shape=(32,32), encoding_dims=100, architecture='2D.WGAN_GP')
        hist = model.train(training_generator=self.datagen,
                           epochs=3)
        self.assertTrue("g_loss" in hist)
        self.assertTrue("d_loss" in hist)

    #-------------------------------------------------#
    #                 Model Inference                 #
    #-------------------------------------------------#
    def test_generate_dcgan(self):
        model = GANNeuralNetwork(channels=3, input_shape=(32,32), encoding_dims=100, architecture='2D.DCGAN')
        
        # create 10 random noise vectors with shape (10, encoding_dims)
        noise = np.random.normal(0, 1, (10, 100))
        preds = model.architecture.generator.predict(noise)

        # check if the output shape is correct (num_images, x_size, y_size, channels)
        self.assertTrue(preds.shape == (10, 32, 32, 3))

    def test_generate_wgan_gp(self):
        # def generator_loss(fake_img):
        #     return -tf.reduce_mean(fake_img)
        
        model = GANNeuralNetwork(channels=3, input_shape=(32,32), encoding_dims=100, architecture='2D.WGAN_GP')
        
        # create 10 random noise vectors with shape (10, encoding_dims)
        noise = np.random.normal(0, 1, (10, 100))
        preds = model.architecture.generator.predict(noise)

        # check if the output shape is correct (num_images, x_size, y_size, channels)
        self.assertTrue(preds.shape == (10, 32, 32, 3))