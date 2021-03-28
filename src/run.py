import os
import sys
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np

# import umap
# import umap.plot


import matplotlib.gridspec as gridspec
# import seaborn as sns
# sns.set(context='notebook', style='white', rc={'figure.figsize':(11,11)})

sys.path.append('../gan-vae-pretrained-pytorch')
from mnist_dcgan.dcgan import Discriminator, Generator
from mnist_classifier.lenet import LeNet5
from torchvision.datasets.mnist import MNIST


batch_size = 2048
latent_size = 100

# load the GAN model
D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netD_epoch_99.pth'))
G.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_dcgan/weights/netG_epoch_99.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

# Load MNIST classifier
classifier = LeNet5().eval()
classifier.load_state_dict(torch.load('../gan-vae-pretrained-pytorch/mnist_classifier/weights/lenet_epoch=12_test_acc=0.991.pth'))
if torch.cuda.is_available():
    classifier = classifier.cuda()
    
    
# Load MNIST dataset
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
(ims, labs) = next(iter(data_loader)) # get whole dataset
if torch.cuda.is_available():
    ims = ims.cuda()
labs = labs.numpy()


def get_perturb_network(latent_size=100, num_classes=0, init_identity=True):
    model = nn.Sequential(
        nn.Linear(latent_size+num_classes, latent_size*2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(latent_size*2, latent_size*2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(latent_size*2, latent_size*2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(latent_size*2, latent_size*2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(latent_size*2, latent_size)
    ).cuda()

    if init_identity:
        # Initialize to identify weight matrix (as if nothing is happening)
        with torch.no_grad():
            for layer in model:
                if type(layer) == nn.Linear:
                    layer.weight = nn.Parameter(torch.eye(latent_size).cuda())
                    layer.bias = nn.Parameter(torch.zeros(latent_size).cuda())
                
    return model



def show_images(ims):
    ims = ims.cpu().detach().numpy()[:25]
    ims = ims.reshape(ims.shape[0], ims.shape[-1], ims.shape[-1])
    R, C = 5, 5
    for i in range(ims.shape[0]):
        plt.subplot(R, C, i + 1)
        plt.imshow(ims[i], cmap='gray')
    plt.show()


def gen_noise(bs, dim=latent_size, class_tensor=None, num_classes=None):
    return torch.randn(bs, dim).cuda()


def gen_images(noise, model=G, disc=D, classifier=classifier, perturb=None, num_classes=0):
    if perturb is not None:
        perturbation = perturb(noise)
        if num_classes > 0:
            noise_p = noise[:,:-num_classes] + perturbation
        else:
            noise_p = noise + perturbation
    else:
        noise_p = noise
    noise_p_lg = noise_p.unsqueeze(-1).unsqueeze(-1)

    images = model(noise_p_lg)
    
    preds = None
    if classifier is not None:
        images_interp = nn.functional.interpolate(images, size=32)
        preds = torch.exp(classifier(images_interp))
        
    disc_score = None
    if disc is not None:
        disc_score = disc(images)
        
    return noise_p, images, preds, disc_score



def generate_sample():
    perturb_one = torch.load('models/one_test.pt')
    
    sample_size=64
    noise = gen_noise(sample_size)
    _, images_perturbed, _, _ = gen_images(noise, perturb=perturb_one)
    images = torch.cat([images_perturbed]*3, dim=1) # add third dimension
    
    grid = torchvision.utils.make_grid(images, nrow=8)
    torchvision.utils.save_image(grid, 'sample_ones.png')
    
    
if __name__ == "__main__":
    generate_sample()