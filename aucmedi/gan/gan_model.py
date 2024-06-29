# External libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import cv2
import os
import csv
import pandas as pd
import json
# Internal libraries/scripts
from aucmedi.gan.gan_architectures.gan_factory import GANFactory

#-----------------------------------------------------#
#           GAN Augmentation (model) class            #
#-----------------------------------------------------#
# Class which represents the GAN Augmentation
class GANNeuralNetwork:
    
    def __init__(self, channels, input_shape=None, architecture='2D.DCGAN',
                  encoding_dims=128, step_channels=64, **kwargs):

        self.input_shape = input_shape
        self.channels = channels
        self.encoding_dims = encoding_dims
        arch_paras = {
            "encoding_dims": encoding_dims,
            "channels": channels,
            "step_channels": step_channels,
            "input_shape": input_shape
        }

        self.gan_factory = GANFactory()
        self.architecture = self.gan_factory.create_model(architecture, **arch_paras, **kwargs)
        self.architecture.compile()

    def compile(self, d_loss_fn, g_loss_fn, d_optimizer, g_optimizer, d_loss_metric, g_loss_metric):
        self.architecture.compile(d_loss_fn, g_loss_fn, d_optimizer, g_optimizer, d_loss_metric, g_loss_metric)

    def train(self, training_generator, epochs=20, callbacks=[], verbose=1, workers=1, use_multiprocessing=False, max_queue_size=10, **kwargs):
        history = self.architecture.fit(training_generator, epochs=epochs, verbose=verbose, callbacks=callbacks, workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size, **kwargs)
        return history.history        

    def generate_images(self, num_images, image_class, save_path, image_format="jpg"):
        noise = np.random.normal(0, 1, (num_images, self.architecture.encoding_dims))
        generated = self.architecture.generate(noise)

        augmented_images = []
        for i in range(num_images):
            filename = f"{image_class}_{i}.{image_format}"
            cv2.imwrite(os.path.join(save_path, filename), generated[i] * 255)
            augmented_images.append(filename)

        return augmented_images
    
    def update_annotation_file(self, interface, annotation_path, augmented_images, image_class, ohe=False, ohe_columns=None):
        if interface == "csv":
            self.update_csv(annotation_path, augmented_images, image_class, ohe, ohe_columns)
        elif interface == "json":
            self.update_json(annotation_path, augmented_images, image_class, ohe, ohe_columns)
        else:
            raise ValueError("Invalid interface type. Supported interfaces are 'csv' and 'json'.")

    def update_csv(self, annotation_path, augmented_images, image_class, ohe=False, ohe_columns=None):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"CSV file not found: {annotation_path}")

        df = pd.read_csv(annotation_path)

        if ohe:
            if ohe_columns is None:
                ohe_columns = [col for col in df.columns if col != 'SAMPLE']
            ohe_data = pd.DataFrame(0, index=np.arange(len(augmented_images)), columns=ohe_columns)
            ohe_data['SAMPLE'] = augmented_images
            ohe_data[image_class] = 1
            df = pd.concat([df, ohe_data], ignore_index=True)
        else:
            new_rows = pd.DataFrame({'SAMPLE': augmented_images, 'CLASS': [image_class] * len(augmented_images)})
            df = pd.concat([df, new_rows], ignore_index=True)

        df.to_csv(annotation_path, index=False)

    def update_json(self, annotation_path, augmented_images, image_class, ohe=False, ohe_columns=None):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"JSON file not found: {annotation_path}")

        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        if ohe:
            if ohe_columns is None:
                raise ValueError("OHE columns must be provided for one-hot encoding format.")
            for img in augmented_images:
                annotations[img] = [1 if col == image_class else 0 for col in ohe_columns]
        else:
            for img in augmented_images:
                annotations[img] = image_class

        with open(annotation_path, 'w') as file:
            json.dump(annotations, file, indent=4)

    def dump(self, generator_path, discriminator_path):
        self.architecture.generator.save(generator_path)
        self.architecture.discriminator.save(discriminator_path)

    def load(self, generator_path, discriminator_path):
        self.architecture.generator = load_model(generator_path)
        self.architecture.discriminator = load_model(discriminator_path)
    