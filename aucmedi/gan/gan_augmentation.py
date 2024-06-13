import numpy as np
import cv2
import os
import csv
import pandas as pd
import json

class GanAugmentation():
    """Interface for data generation module."""
    def __init__(self, gan_model):
        self.gan_model = gan_model

    def generate_images(self, num_images, image_class, save_path, image_format="jpg"):
        noise = np.random.normal(0, 1, (num_images, self.gan_model.encoding_dims))
        generated = self.gan_model.generate(noise)

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
