import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

class DCGAN(tf.keras.Model):
    def __init__(self, discriminator=None, generator=None, input_shape=(28, 28), channels=1, step_channels=64, encoding_dims=128, **kwargs):
        super(DCGAN, self).__init__()
        self.encoding_dims = encoding_dims
        self.latent_dim = encoding_dims
        self.image_shape = input_shape + (channels,)
        self.step_channels = step_channels

        if discriminator is None:
            self.discriminator = self.build_discriminator()
        else:
            self.discriminator = discriminator

        if generator is None:
            self.generator = self.build_generator()
        else:
            self.generator = generator

    def compile(self, d_optimizer=Adam(), g_optimizer=Adam(), loss_fn=tf.keras.losses.BinaryCrossentropy()):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")   
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def build_generator(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                layers.Dense(4 * 4 * 256),
                layers.Reshape((4, 4, 256)),
                layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(3, kernel_size=3, padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        model.summary()
        return model

    def build_discriminator(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        model.summary()
        return model

    def train_step(self, data):
        real_images, _ = data
        batch_size = tf.shape(real_images)[0]
        print(f"Batch size: {batch_size.numpy()}")

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        print(f"Random latent vectors shape: {random_latent_vectors.shape}")

        generated_images = self.generator(random_latent_vectors)
        print(f"Generated images shape: {generated_images.shape}")

        combined_images = tf.concat([generated_images, real_images], axis=0)
        print(f"Combined images shape: {combined_images.shape}")

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        print(f"Labels shape: {labels.shape}")

        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        print(f"Labels after noise addition: {labels.numpy()}")

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            print(f"Discriminator predictions: {predictions.numpy()}")

            d_loss = self.loss_fn(labels, predictions)
            print(f"Discriminator loss: {d_loss.numpy()}")

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        for grad, weight in zip(grads, self.discriminator.trainable_weights):
            print(f"Weight shape: {weight.shape}, Gradient shape: {None if grad is None else grad.shape}")

        if any(g is None for g in grads):
            raise ValueError("Discriminator gradients are None. Check the model and loss function.")

        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        misleading_labels = tf.zeros((batch_size, 1))
        print(f"Misleading labels shape: {misleading_labels.shape}")

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(generated_images)
            print(f"Generator predictions: {predictions.numpy()}")

            g_loss = self.loss_fn(misleading_labels, predictions)
            print(f"Generator loss: {g_loss.numpy()}")

        grads = tape.gradient(g_loss, self.generator.trainable_weights)

        for grad, weight in zip(grads, self.generator.trainable_weights):
            print(f"Weight shape: {weight.shape}, Gradient shape: {None if grad is None else grad.shape}")

        if any(g is None for g in grads):
            raise ValueError("Generator gradients are None. Check the model and loss function.")

        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def fit(self, training_generator, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step in range(len(training_generator)):
                data = next(iter(training_generator))
                print(f"Step {step + 1}/{len(training_generator)} - Data sample shape: {data[0].shape}, Data label shape: {data[1].shape}")
                self.train_step(data)
            print(f"Completed epoch {epoch + 1}")

# Example usage
# dcgan = DCGAN()
# dcgan.compile()
# dcgan.fit(training_data, epochs=50)
