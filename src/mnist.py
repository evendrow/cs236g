import sys
import torch
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

sys.path.append('../gan-vae-pretrained-pytorch')
from mnist_dcgan.dcgan import Discriminator, Generator
from mnist_classifier.lenet import LeNet5

def load_mnist_gan(device='cuda'):
    """ 
    Loads MNIST GAN model

    Arguments:
    device -- device to which the model will be moved

    Returns:
    (G, D) -- generator and discriminator model

    """
    D = Discriminator(ngpu=1).eval()
    G = Generator(ngpu=1).eval()

    D.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netD_epoch_99.pth', map_location=device))
    G.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netG_epoch_99.pth', map_location=device))
    
    G.to(device)
    D.to(device)

    return (G, D)


def load_mnist_classifier(device='cuda'):
    """ 
    Loads MNIST digit classifier model

    Arguments:
    device -- device to which the model will be moved

    """
    classifier = LeNet5().eval()
    classifier.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_classifier/weights/lenet_epoch=12_test_acc=0.991.pth'))
    classifier.to(device)
    return classifier


def load_mnist_dataset():
    """ Loads MNIST dataset 

    Images are transformed to size 28x28 and normalized.

    Returns:
    dataloader -- torch DataLoader object
    """

    # Weird hack to get mnist loading to work prooperly
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    data_root = '~/mnist'
    data_test = MNIST(data_root,
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]))
    data_loader = torch.utils.data.DataLoader(data_test, 
                                              batch_size=10000,
                                              shuffle=False)

    return data_loader


class MNISTFeatureExtractor:
    """ Uses an intermediate layer in an MNIST classifier to
        perform feature extraction, especially for visualization """

    def __init__(self, device='cuda'):
        
        self.device = device

        # We add a hook to an intermediate layer to log its outputs
        self.feat_extractor = load_mnist_classifier(device=device)
        self.feat_extractor.fc.f6.register_forward_hook(self._hook)

        # Outputs will be appended to this list
        self.fc = []
        
    def _hook(self, module, input, output):
        self.fc.append(output)
        
    def get_feats(self, images):
        # Interpolate input image if needed
        if images.shape[-1] != 32:
            images = nn.functional.interpolate(ims, size=32)

        images.to(self.device)

        # Reset output list and log features
        self.fc = []
        with torch.no_grad():
            self.feat_extractor(ims)
            
        return self.fc[0]