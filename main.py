import os

import torch
import matplotlib.pyplot as plt

import argparse

from src.train import train
from src.models import PerturbationNetwork
from src.mnist import load_mnist_gan, load_mnist_classifier, load_mnist_dataset, mnist_repeat, MNISTGenerator, MNISTClassifier, MNISTFeatureExtractor
from src.utils import create_logger, load_default_config, load_config
from src.vis import get_umap_embedding, generate_samples



def run_experiment(cfg):

    # Create experiment folder
    if not os.path.exists(cfg["log_dir"]):
        os.makedirs(cfg["log_dir"])

    exp_dir = os.path.join(cfg["log_dir"], cfg['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if not torch.cuda.is_available():
        cfg["device"] = 'cpu'
        
    logger = create_logger(exp_dir, phase='train')
    logger.info(f'Device -> ' + str(cfg["device"]))
    

    # cudnn related setting
    torch.backends.cudnn.benchmark = cfg['CUDNN']['benchmark']
    torch.backends.cudnn.deterministic = cfg['CUDNN']['deterministic']
    torch.backends.cudnn.enabled = cfg['CUDNN']['enabled']

    # Create argument dict to feed into train method
    args = dict()
    args["device"]         = cfg["device"]
    args["batch_size"]     = cfg["batch_size"]
    args["latent_dim"]     = cfg["latent_dim"]
    args["class_weight"]   = cfg["class_weight"]
    args["perturb_weight"] = cfg["perturb_weight"]
    args["disc_weight"]    = cfg["disc_weight"]
    args["iters"]          = cfg["iters"]
    args["log_steps"]      = cfg["log_steps"]
    args["log_dir"]        = exp_dir

    # <================== Initialize Models

    perturb_net = PerturbationNetwork(latent_dim=cfg["latent_dim"], 
                                      num_hidden_layers=cfg["perturb_cfg"]["hidden_layers"],
                                      dropout=cfg["perturb_cfg"]["dropout"])
    perturb_net.to(device=cfg["device"])

    G = MNISTGenerator(device=cfg["device"])
    classifier = MNISTClassifier().to(cfg["device"])
    feature_extractor = MNISTFeatureExtractor(device=cfg["device"])

    target = torch.Tensor(cfg["target"]).to(cfg["device"])
    target = target / target.sum()
    logger.info(f"Target: {str(target.cpu().numpy())}")

    logger.info(f'Initialized models.')

    # <================== Load Dataset

    data_loader = load_mnist_dataset()    
    (ims, labs) = next(iter(data_loader))
    ims = ims.to(cfg["device"])
    labs = labs.to(cfg["device"])

    logger.info(f'Loaded annotations.')


    # <================== Visualization

    if cfg["vis"]["do_vis"]:
        mapping, real_embeds = get_umap_embedding(ims, feature_extractor)

        args["save_dir"]          = os.path.join(exp_dir, 'frames')
        args["vis_steps"]         = cfg["vis"]["vis_steps"]
        args["vis_points"]        = cfg["vis"]["vis_points"]
        args["feature_extractor"] = feature_extractor
        args["mapping"]           = mapping
        args["real_embeds"]       = real_embeds

    # <================== Train

    losses = train(perturb_net, G, classifier, target, logger, **args)

    # <================== Save figure and checkpoint

    # Model Checkpoint
    torch.save(perturb_net, os.path.join(exp_dir, 'model.pt'))

    # Loss plot
    plt.figure(figsize=(10,6), dpi=256)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(exp_dir, 'loss.png'))

    

    # Images for generated distribution
    perturb_net.eval()
    with torch.no_grad():
        images = generate_samples(cfg['hist_sample_size'], cfg["latent_dim"], G, batch_size=cfg['batch_size'], perturb=perturb_net, device=cfg["device"])

    classes = classifier(images)
    class_nums = classes.argmax(-1)

    bins = list(range(10))

    plt.figure(figsize=(5, 5), dpi=256)
    plt.hist(class_nums.detach().cpu().numpy(), bins=bins, density=True)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(exp_dir, 'hist.png'))


    # Desired distribution
    desired_dist = mnist_repeat(target, cfg['hist_sample_size'])
    plt.figure(figsize=(5, 5), dpi=256)
    plt.hist(desired_dist.cpu().numpy(), bins=bins, density=True)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(exp_dir, 'desired_hist.png'))

    logger.info("All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    cfg_file = args.cfg
    print('Got config file', cfg_file)

    if cfg_file is not None:
        cfg = load_config(cfg_file)
    else:
        cfg = load_default_config()

    print(cfg, end='\n\n')

    run_experiment(cfg)
