import numpy as np
import cv2
import os

class GanAugmentation():
    def __init__(self, gan_model, num_images, image_class, save_path, image_format):
        self.gan_model = gan_model
        self.num_images = num_images
        self.image_class = image_class
        self.save_path = save_path
        self.image_format = image_format.lower()

        # Ensure save_path exists
        os.makedirs(self.save_path, exist_ok=True)

    def generate_images(self):
        """ Generates augmented images using the GAN model and saves them to disk.

        Returns:
            generated_images (list): A list of generated images.
        """
        # Generate random noise as input to the GAN model
        noise = np.random.normal(0, 1, (self.num_images, 100))
        # Generate images using the GAN model
        generated_images = self.gan_model.generate(noise)
        
        # Apply augmentation to the generated images
        augmented_images = []
        augmented_images_filename = []
        for idx, image in enumerate(generated_images):
            augmented_images.append(image)

            # Save the augmented image
            filename = f"{self.image_class}_{idx}.{self.image_format}"
            cv2.imwrite(os.path.join(self.save_path, filename), image * 255)

            augmented_images_filename.append(filename)

        return augmented_images_filename


