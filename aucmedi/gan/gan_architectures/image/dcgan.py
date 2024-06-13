
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

import numpy as np

# Internal libraries
from aucmedi.gan.gan_architectures import GAN_Architecture_Base


#-----------------------------------------------------#
#                 DCGAN Architecture                #
#-----------------------------------------------------#
class DCGAN(GAN_Architecture_Base):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, encoding_dims=100, channels=1, input_shape=(32,32), step_channels=64, optimizer=Adam(0.0002, 0.5), metrics=['accuracy'], loss='binary_crossentropy'): 
        super().__init__(encoding_dims, channels, optimizer, metrics, loss, input_shape, step_channels)
        self.build_and_compile()

    def build_generator(self):
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
    
    def build_discriminator(self):
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
    
    def build_and_compile(self):
        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.encoding_dims,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    
    def train(self, training_generator, epochs):
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

