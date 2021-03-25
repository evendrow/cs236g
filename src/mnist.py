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

    D.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netD_epoch_99.pth'))
    G.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netG_epoch_99.pth'))
    
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