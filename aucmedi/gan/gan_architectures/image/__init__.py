#-----------------------------------------------------#
#                    Architectures                    #
#-----------------------------------------------------#
# Vanilla Classifier
from aucmedi.gan.gan_architectures.image.dcgan import DCGAN
from aucmedi.gan.gan_architectures.image.wgan_gp import WGAN_GP

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#

# Architecture Dictionary
architecture_dict = {
    "DCGAN": DCGAN,
    "WGAN_GP": WGAN_GP,
}