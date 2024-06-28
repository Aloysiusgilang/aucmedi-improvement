import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from aucmedi.gan.gan_architectures.arch_base import GAN_Architecture_Base

class WGAN_GP(GAN_Architecture_Base):
    def __init__(self, input_shape=(64, 64), 
                 channels=1, step_channels=64, encoding_dims=128, gp_weight=10, discriminator_extra_steps=5):
        super(WGAN_GP, self).__init__( 
            encoding_dims=encoding_dims,
            channels=channels,
            input_shape=input_shape,
            step_channels=step_channels
        )
        self.gp_weight = gp_weight
        self.d_steps = discriminator_extra_steps

    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)


    def compile(self, d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
                g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
                d_loss_fn=discriminator_loss, g_loss_fn=generator_loss,
                d_loss_metric=tf.metrics.Mean(name='d_loss'), g_loss_metric=tf.metrics.Mean(name='g_loss')):
        super(WGAN_GP, self).compile(d_loss_fn, g_loss_fn, d_optimizer, g_optimizer, d_loss_metric, g_loss_metric)
    

    def build_generator(self):
        num_repeats = self.image_shape[0].bit_length() - 4
        d = self.step_channels * (2 ** num_repeats)

        # Initialize input layer
        model_input = keras.Input(shape=self.encoding_dims)
        x = layers.Dense(d * 4 * 4)(model_input)
        x = layers.Reshape((4, 4, d))(x)

        for i in range(num_repeats):
            x = layers.Conv2DTranspose(d // 2, (4, 4), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            d = d // 2
        
        x = layers.Conv2DTranspose(self.image_shape[2], (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)
        model_output = x

        return Model(inputs=model_input, outputs=model_output)
    
    def build_discriminator(self):
        num_repeats = self.image_shape[0].bit_length() - 4
        d = self.step_channels

        # Initialize input layer
        model_input = keras.Input(shape=self.image_shape)
        x = layers.Conv2D(d, (4, 4), strides=(2, 2), padding='same')(model_input)
        x = layers.LeakyReLU(alpha=0.2)(x)

        for i in range(num_repeats):
            x = layers.Conv2D(d * 2, (4, 4), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            d = d * 2

        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        return Model(inputs=model_input, outputs=x)
    
    def gradient_penalty(self, real_images, fake_images):

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)  
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            #1. Get the discriminator output for this interpolated image
            pred = self.discriminator(interpolated, training=True)
        
        #2. Calculate the gradients w.r.t to this interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        #3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, _ = data
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.encoding_dims))

            with tf.GradientTape() as tape:
                # generate fake images
                fake_images = self.generator(random_latent_vectors, training=True)

                #get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)

                #get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                #calculate the discriminator loss
                d_cost = self.d_loss_fn(real_logits, fake_logits)

                #calculate the gradient penalty
                gp = self.gradient_penalty(real_images, fake_images)

                #add the gradient penalty to the discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            #get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

            #update the discriminator
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        
        # train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.encoding_dims))
        with tf.GradientTape() as tape:
            # generate fake images
            fake_images = self.generator(random_latent_vectors, training=True)

            #get the logits for the fake images
            fake_logits = self.discriminator(fake_images, training=True)

            #calculate the generator loss
            g_loss = self.g_loss_fn(fake_logits)

        #get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        #update the generator
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}