import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from aucmedi.gan.gan_architectures.arch_base import GAN_Architecture_Base

class DCGAN(GAN_Architecture_Base):
    def __init__(self, input_shape=(64, 64), channels=1, step_channels=64, encoding_dims=128):
        super(DCGAN, self).__init__( 
            encoding_dims=encoding_dims,
            channels=channels,
            input_shape=input_shape,
            step_channels=step_channels
        )

    def compile(self, d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
                g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
                d_loss_fn=keras.losses.BinaryCrossentropy(), g_loss_fn=keras.losses.BinaryCrossentropy(),
                d_loss_metric=keras.metrics.Mean(name='d_loss'), g_loss_metric=keras.metrics.Mean(name='g_loss')):
        super(DCGAN, self).compile(d_loss_fn, g_loss_fn, d_optimizer, g_optimizer, d_loss_metric, g_loss_metric)

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

    def train_step(self, data):
        real_images, _ = data
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.encoding_dims))

        generated_images = self.generator(random_latent_vectors)

        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)

            d_loss = self.d_loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        if any(g is None for g in grads):
            raise ValueError("Discriminator gradients are None. Check the model and loss function.")

        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(generated_images)

            g_loss = self.d_loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)

        if any(g is None for g in grads):
            raise ValueError("Generator gradients are None. Check the model and loss function.")

        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}
