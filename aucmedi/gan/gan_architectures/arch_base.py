from abc import ABC, abstractmethod

class GAN_Architecture_Base(ABC):
    
    @abstractmethod
    def __init__(self, encoding_dims, channels, optimizer, metrics, loss, input_shape, step_channels):
        self.input = input_shape + (channels,)
        self.encoding_dims = encoding_dims
        self.step_channels = step_channels
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.discriminator = None
        self.generator = None
        self.combined = None

    @abstractmethod
    def build_generator(self):
        pass
    
    @abstractmethod
    def build_discriminator(self):
        pass

    @abstractmethod
    def build_and_compile(self):
        pass
    
    @abstractmethod
    def train(self, training_generator, epochs):
        pass

