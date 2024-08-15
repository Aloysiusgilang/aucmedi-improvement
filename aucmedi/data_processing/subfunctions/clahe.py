#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
# External libraries
import albumentations
import numpy as np
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
class Clahe(Subfunction_Base):
    """ A CLAHE Subfunction class which applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    ???+ info "2D image"
        CLAHE is applied using the `CLAHE` transform from the `albumentations` library. <br>
        https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE

    ???+ info "3D volume"
        CLAHE is not supported for 3D volumes in the `albumentations` library.

    Args:
        clip_limit (float):         Threshold for contrast limiting. Default is 4.0.
        tile_grid_size (tuple):     Size of grid for histogram equalization. Default is (8, 8).
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        """ Initialization function for creating a CLAHE Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            clip_limit (float):         Threshold for contrast limiting. Default is 4.0.
            tile_grid_size (tuple):     Size of grid for histogram equalization. Default is (8, 8).
        """
        # Initialize parameter
        params = {"p":1.0, "always_apply":True, "clip_limit":clip_limit, "tile_grid_size":tile_grid_size}
        # Initialize CLAHE transform
        self.aug_transform = albumentations.CLAHE(**params)

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Determine the dimensions of the image
        if image.ndim == 3:
            # 2D image
            image_uint8 = (image).astype(np.uint8)
            augmented = self.aug_transform.apply(img=image_uint8)
            transformed_image = augmented.astype(np.float64)

        elif image.ndim == 4:
            # 3D image
            image_uint8 = (image).astype(np.uint8)
            transformed_image = np.zeros_like(image, dtype=np.float64)
            for i in range(image.shape[0]):
                augmented = self.aug_transform.apply(img=image_uint8[i])
                transformed_image[i] = augmented.astype(np.float64)
        
        else:
            raise ValueError("Unsupported image dimensions")
        
        return transformed_image