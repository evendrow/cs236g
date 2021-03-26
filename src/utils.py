import os
import logging
import yaml

import torch
import torch.nn as nn

def path_to_repo(*args): # REPO/arg1/arg2
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

def path_to_src(*args): # REPO/src/arg1/arg2
    return path_to_repo("src", *args)

def path_to_conf(*args): # REPO/src/configs/arg1/arg2
    return path_to_src("configs", *args)


def gen_noise(bs, dim, device='cuda'):
    return torch.randn(bs, dim).to(device)

def gen_images(noise, model, disc, classifier, perturb=None):
    if perturb is not None:
        noise = perturb(noise)

    noise = noise
    images = model(noise.unsqueeze(-1).unsqueeze(-1))
    
    preds = None
    if classifier is not None:
        images_interp = nn.functional.interpolate(images, size=32)
        preds = torch.exp(classifier(images_interp))
        
    disc_score = None
    if disc is not None:
        disc_score = disc(images)
        
    return noise, images, preds, disc_score


PATH_TO_DEFAULT = path_to_conf("default.yaml")

def load_default_config():
    return load_config(PATH_TO_DEFAULT)

def load_config(path):
    """
    Load the config file and make any dynamic edits.
    """
    with open(path, "rt") as reader:
        config = yaml.load(reader, Loader=yaml.Loader)

    # config["experiment"] = os.path.splitext(os.path.basename(path))[0]
    # config["logs_dir"], config["ckpt_dir"], config["runs_dir"] = init_output_dirs(config["experiment"])
        
    return config


def create_logger(logdir, phase='train'):
    head = '%(asctime)-15s %(message)s'
    if logdir != '':
        log_file = os.path.join(logdir, f'{phase}_log.txt')
        logging.basicConfig(filename=log_file, format=head)
    else:
        logging.basicConfig(format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logdir != '':
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    """
    From https://github.com/mkocabas/VIBE/blob/master/lib/core/trainer.py
    Keeps track of a moving average.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.mov_avg = 0

    def update(self, val, n=1, a=0.9):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.mov_avg = a*self.mov_avg + (1-a)*val