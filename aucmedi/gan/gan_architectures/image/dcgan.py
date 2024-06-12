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
#                    Documentation                    #
#-----------------------------------------------------#
""" The classification variant of the Vanilla architecture.

No intensive hardware requirements, which makes it ideal for debugging.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.Vanilla"               |
| Input_shape              | (224, 224)                 |
| Standardization          | "z-score"                  |

???+ abstract "Reference - Implementation"
    [https://github.com/wanghsinwei/isic-2019/](https://github.com/wanghsinwei/isic-2019/) <br>
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model
import numpy as np
# Internal libraries
from aucmedi.neural_network.gan_architectures import GAN_Architecture_Base


#-----------------------------------------------------#
#                 Vanilla Architecture                #
#-----------------------------------------------------#
class DCGAN(GAN_Architecture_Base):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, encoding_dims=100, channels=1, input_shape=(32,32), step_channels=64): 
        self.input = input_shape + (channels,)
        self.encoding_dims = encoding_dims
        self.step_channels = step_channels

    def create_generator(self):
        num_repeats = self.input[0].bit_length() - 4
        d = self.step_channels * (2 ** num_repeats)
        print('num_repeats',num_repeats)

        # Initialize input layer
        model_input = Input(shape=self.encoding_dims)
        x = Dense(d * 4 * 4)(model_input)
        x = Reshape((4, 4, d))(x)

        for i in range(num_repeats):
            x = Conv2DTranspose(d // 2, (4, 4), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            d = d // 2
        
        x = Conv2DTranspose(self.input[2], (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)
        model_output = x

        return Model(inputs=model_input, outputs=model_output)
    
    def create_discriminator(self):
        num_repeats = self.input[0].bit_length() - 4
        d = self.step_channels
        print('num_repeats',num_repeats)

        # Initialize input layer
        model_input = Input(shape=self.input)
        x = Conv2D(d, (4, 4), strides=(2, 2), padding='same')(model_input)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(num_repeats):
            x = Conv2D(d * 2, (4, 4), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            d = d * 2

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=model_input, outputs=x)
    
    def create_model(self):
        generator = self.create_generator()
        discriminator = self.create_discriminator()

        print('Generator', generator.summary())
        print('Discriminator', discriminator.summary())

        return generator, discriminator
    

        
