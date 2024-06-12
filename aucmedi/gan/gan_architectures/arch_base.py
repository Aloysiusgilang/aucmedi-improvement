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

    @abstractmethod
    def build_generator(self):
        return None
    
    @abstractmethod
    def build_discriminator(self):
        return None
    
    @abstractmethod
    def train(self, training_generator, epochs):
        return None

