import albumentations
import numpy as np
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

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
# Internal libraries/scripts

#-----------------------------------------------------#
#              Subfunction class: Blur                #
#-----------------------------------------------------#
class MedianBlur(Subfunction_Base):
    """ A Blur Subfunction class which applies blur to an image.

    ???+ info "2D image"
        Blur is applied using the `Blur` transform from the `albumentations` library. <br>
        https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur

    ???+ info "3D volume"
        Blur is not supported for 3D volumes in the `albumentations` library.

    Args:
        blur_limit (int):           Maximum size of the blur kernel. Default is 7.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, blur_limit=7):
        """ Initialization function for creating a Blur Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            blur_limit (int):           Maximum size of the blur kernel. Default is 7.
        """
        # Initialize parameter
        params = {"p":1.0, "always_apply":True, "blur_limit":blur_limit}
        # Initialize Blur transform
        self.aug_transform = albumentations.MedianBlur(**params)

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Convert float64 image to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply Blur augmentation
        augmented = self.aug_transform(image=image_uint8)
        
        # Convert image back to float64
        transformed_image = augmented['image'].astype(np.float64) / 255.0
        return transformed_image
