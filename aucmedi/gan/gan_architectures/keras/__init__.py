#-----------------------------------------------------#
#                    Architectures                    #
#-----------------------------------------------------#
# Vanilla Classifier
from aucmedi.gan.gan_architectures.keras.dcgan import DCGAN

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#

# Architecture Dictionary
architecture_dict = {
    "DCGAN": DCGAN
}
# List of implemented architectures
architectures = list(architecture_dict.keys())