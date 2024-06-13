from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np
from functools import partial


from aucmedi.gan.gan_architectures import GAN_Architecture_Base

class RandomWeightedAverage(tf.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated image samples"""
    def call(self, inputs):
        alpha = tf.random.uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


#-----------------------------------------------------#
#                 WGAN-GP Architecture                #
#-----------------------------------------------------#
class WGAN_GP(GAN_Architecture_Base):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, encoding_dims=100, channels=1, input_shape=(32,32), step_channels=64, optimizer=RMSprop(lr=0.00005), metrics=['accuracy'], loss='binary_crossentropy'): 
        super().__init__(encoding_dims, channels, optimizer, metrics, loss, input_shape, step_channels)
        self.n_discriminators = 5
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
        # build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator.trainable = False

        # image input (real sample)
        real_img = Input(shape=self.input)

        # noise input
        z_disc = Input(shape=(self.encoding_dims,))

        # generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        validity_interpolated = self.discriminator(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])

        self.discriminator_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss], optimizer=self.optimizer, loss_weights=[1, 1, 10])

        # for the generator we freeze the discriminator's layer
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.encoding_dims,))
        img = self.generator(z_gen)

        # Discriminator determines validity
        valid = self.discriminator(img)

        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def gradient_penalty(self, real_imgs, fake_imgs):
        alpha = tf.random.uniform(shape=[real_imgs.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated_imgs = alpha * real_imgs + (1 - alpha) * fake_imgs

        with tf.GradientTape() as tape:
            tape.watch(interpolated_imgs)
            interpolated_validity = self.discriminator(interpolated_imgs)

        gradients = tape.gradient(interpolated_validity, interpolated_imgs)
        gradients_norm = tf.norm(tf.reshape(gradients, [gradients.shape[0], -1]), axis=1)
        gradient_penalty = tf.reduce_mean((gradients_norm - 1) ** 2)

        return gradient_penalty
    
    def train(self, training_generator, epochs):
        for epoch in range(epochs):
            for i in range(len(training_generator)):
                real_imgs, labels = training_generator[i]
                current_batch_size = len(real_imgs)
                valid = -np.ones((current_batch_size, 1))
                fake = np.ones((current_batch_size, 1))
                dummy = np.zeros((current_batch_size, 1))

                for _ in range(self.n_discriminators):
                    noise = np.random.normal(0, 1, (current_batch_size, self.encoding_dims))
                    d_loss = self.discriminator_model.train_on_batch([real_imgs, noise], [valid, fake, dummy])
                
                g_loss = self.generator_model.train_on_batch(noise, valid)

                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        


                        
                

