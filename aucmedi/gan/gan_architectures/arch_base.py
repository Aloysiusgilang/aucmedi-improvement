import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from aucmedi.gan.gan_architectures.metric.kid import KID

class GAN_Architecture_Base(tf.keras.Model):
    
    def __init__(self, encoding_dims, channels, input_shape, step_channels, batch_size=32):
        super(GAN_Architecture_Base, self).__init__()
        self.encoding_dims = encoding_dims
        self.image_shape = input_shape + (channels,)
        self.step_channels = step_channels
        self.batch_size = batch_size
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

    def compile(self, d_loss_fn, g_loss_fn, d_optimizer, g_optimizer, d_loss_metric, g_loss_metric):
        super(GAN_Architecture_Base, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_metric = d_loss_metric   
        self.g_loss_metric = g_loss_metric
        self.kid = KID(self.image_shape[0])

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.kid]
    
    def build_generator(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def build_discriminator(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def set_generator(self, generator):
        self.generator = generator
    
    def set_discriminator(self, discriminator):
        self.discriminator = discriminator
    
    @tf.function
    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method")
    
    def test_step(self, real_images):
        latent_samples = tf.random.normal(shape=(self.batch_size, self.encoding_dims))
        generated_images = self.generator(latent_samples, training=False)

        self.kid.update_state(real_images, generated_images)

        # only KID is measured during the evaluation phase for computational efficiency
        return {self.kid.name: self.kid.result()}
    
    def generate(self, noise):
        return self.generator(noise)

