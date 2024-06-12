
#-----------------------------------------------------#
#               General library imports               #
#-----------------------------------------------------#
# Abstract Base Class for Architectures
from aucmedi.gan.gan_architectures.arch_base import GAN_Architecture_Base

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Initialize combined architecture_dict for image & volume architectures
architecture_dict = {}

# Add image architectures to architecture_dict
from aucmedi.gan.gan_architectures.image import architecture_dict as arch_image
for arch in arch_image:
    architecture_dict["2D." + arch] = arch_image[arch]

