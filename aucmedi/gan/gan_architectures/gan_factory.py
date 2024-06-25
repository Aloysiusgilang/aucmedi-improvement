# gan factory 
from aucmedi.gan.gan_architectures.image import architecture_dict as arch_image
from aucmedi.gan.gan_architectures.arch_base import GAN_Architecture_Base

class GANFactory():
    def __init__(self) :
        self.architecture_dict = {}
        self.load_architectures()

    def load_architectures(self):
        for arch in arch_image:
            self.architecture_dict["2D." + arch] = arch_image[arch]

    def create_model(self, arch_name, **kwargs):
        if isinstance(arch_name, str) and arch_name in self.architecture_dict:
            return self.architecture_dict[arch_name](**kwargs)
        elif isinstance(arch_name, GAN_Architecture_Base):
            return arch_name
        else:
            raise ValueError("Architecture not found: " + arch_name)
        
        

