import sys
import math

import torch
from torchvision import utils

from src.models import PerturbGenerator

sys.path.append('../style-based-gan-pytorch')
from model import StyledGenerator

def load_stylegan_generator(device='cuda', path='../../style-based-gan-pytorch/checkpoint/stylegan-128px-running-100000.model'):
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(path))
    generator.eval()
    
    return generator

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

def gen_stylegan_noise(batch_size, device='cuda'):
    noise = torch.randn(batch_size, 512).to(device)
    return noise

def gen_stylegan_images(generator, noise, device='cuda'):
    mean_style = get_mean_style(g, device)
    
    SIZE = 128
    step = int(math.log(SIZE, 2)) - 2
    
    images = generator(
        noise,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    return images

class StyleGANGenerator(PerturbGenerator):
    
    def __init__(self, device='cuda'):
        super().__init__(device=device)
        self.g = load_stylegan_generator(device=device)
        self.mean_style = get_mean_style(self.g, device=device)
        self.SIZE = 128
        self.step = int(math.log(self.SIZE, 2)) - 2
        
    def get_latent_size(self):
        return 512
    
    def gen_noise(self, batch_size):
        noise = torch.randn(batch_size, 512).to(self.device)
        return noise
    
    def gen_samples(self, noise):
        images = self.g(
            noise,
            step=self.step,
            alpha=1,
            mean_style=self.mean_style,
            style_weight=0.7,
        )
        return images