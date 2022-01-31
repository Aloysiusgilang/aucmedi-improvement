#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import unittest
import numpy as np
import random
#Internal libraries
from aucmedi import Image_Augmentation, Volume_Augmentation

#-----------------------------------------------------#
#             Unittest: Image Augmentation            #
#-----------------------------------------------------#
class AugmentationTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        random.seed(1234)
        np.random.seed(1234)
        # Create Grayscale data
        img_gray_2D = np.random.rand(16, 16, 1) * 254
        self.imgGRAY2d = np.float32(img_gray_2D)
        img_gray_3D = np.random.rand(16, 16, 16, 1) * 254
        self.imgGRAY3d = np.float32(img_gray_3D)
        # Create RGB data
        img_rgb_2D = np.random.rand(16, 16, 3) * 254
        self.imgRGB2d = np.float32(img_rgb_2D)
        img_rgb_3D = np.random.rand(16, 16, 16, 3) * 254
        self.imgRGB3d = np.float32(img_rgb_3D)

    #-------------------------------------------------#
    #               Image Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_IMAGE_create(self):
        data_aug = Image_Augmentation()
        self.assertIsInstance(data_aug, Image_Augmentation)

    # Application
    def test_IMAGE_application(self):
        data_aug = Image_Augmentation(flip=True, rotate=True,
                     brightness=True, contrast=True, saturation=True,
                     hue=True, scale=True, crop=True, grid_distortion=True,
                     compression=True, gaussian_noise=True,
                     gaussian_blur=True, downscaling=True, gamma=True,
                     elastic_transform=True)
        data_aug.aug_crop_shape = (8, 8)
        data_aug.build()
        img_augGRAY = data_aug.apply(self.imgGRAY2d)
        img_augRGB = data_aug.apply(self.imgRGB2d)
        self.assertFalse(np.array_equal(img_augGRAY, self.imgGRAY2d))
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB2d))

    # Rebuild Augmentation Operator
    def test_IMAGE_rebuild(self):
        data_aug = Image_Augmentation(flip=False, rotate=False,
                     brightness=False, contrast=False, saturation=False,
                     hue=False, scale=False, crop=False, grid_distortion=False,
                     compression=False, gaussian_noise=False,
                     gaussian_blur=False, downscaling=False, gamma=False,
                     elastic_transform=False)
        img_augRGB = data_aug.apply(self.imgRGB2d)
        self.assertTrue(np.array_equal(img_augRGB, self.imgRGB2d))
        data_aug.aug_flip = True
        data_aug.aug_flip_p = 1.0
        data_aug.build()
        img_augRGB = data_aug.apply(self.imgRGB2d)
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB2d))

    #-------------------------------------------------#
    #               Volume Functionality              #
    #-------------------------------------------------#
    # Class Creation
    def test_VOLUME_create(self):
        data_aug = Volume_Augmentation()
        self.assertIsInstance(data_aug, Volume_Augmentation)

    # Application
    def test_VOLUME_application(self):
        data_aug = Volume_Augmentation(flip=True, rotate=True,
                     brightness=True, contrast=True, saturation=True,
                     hue=True, scale=True, crop=True, grid_distortion=True,
                     compression=True, gaussian_noise=True,
                     gaussian_blur=True, downscaling=True, gamma=True,
                     elastic_transform=True)
        data_aug.aug_crop_shape = (8, 8, 8)
        data_aug.build()
        img_augGRAY = data_aug.apply(self.imgGRAY3d)
        img_augRGB = data_aug.apply(self.imgRGB3d)
        self.assertFalse(np.array_equal(img_augGRAY, self.imgGRAY3d))
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB3d))

    # Rebuild Augmentation Operator
    def test_VOLUME_rebuild(self):
        data_aug = Volume_Augmentation(flip=False, rotate=False,
                     brightness=False, contrast=False, saturation=False,
                     hue=False, scale=False, crop=False, grid_distortion=False,
                     compression=False, gaussian_noise=False,
                     gaussian_blur=False, downscaling=False, gamma=False,
                     elastic_transform=False)
        img_augRGB = data_aug.apply(self.imgRGB3d)
        self.assertTrue(np.array_equal(img_augRGB, self.imgRGB3d))
        data_aug.aug_flip = True
        data_aug.aug_flip_p = 1.0
        data_aug.build()
        img_augRGB = data_aug.apply(self.imgRGB3d)
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB3d))
