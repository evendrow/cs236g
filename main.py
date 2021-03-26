import os

import torch
import matplotlib.pyplot as plt

from src.train import train
from src.models import PerturbationNetwork
from src.mnist import load_mnist_gan, load_mnist_classifier, load_mnist_dataset, MNISTClassifier, MNISTFeatureExtractor
from src.utils import load_default_config
from src.vis import get_umap_embedding, generate_samples

def run_experiment(cfg):

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
    args["log_dir"]        = cfg["log_dir"]

    # <================== Initialize Models

    perturb_net = PerturbationNetwork(latent_dim=cfg["latent_dim"], 
                                      num_hidden_layers=cfg["perturb_cfg"]["hidden_layers"],
                                      dropout=cfg["perturb_cfg"]["dropout"])
    perturb_net.to(device=cfg["device"])

    G, D = load_mnist_gan(device=cfg["device"])
    classifier = MNISTClassifier().to(cfg["device"])
    feature_extractor = MNISTFeatureExtractor(device=cfg["device"])

    target = torch.Tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(cfg["device"])

    # logger.info('Initialized models.')

    # <================== Load Dataset

    data_loader = load_mnist_dataset()    
    (ims, labs) = next(iter(data_loader))

    # logger.info(f'Loaded {len(dataset)} annotations.')


    # <================== Visualization

    if cfg["vis"]["save_dir"] != '':
        mapping, real_embeds = get_umap_embedding(ims[:10], feature_extractor)

        args["save_dir"]          = cfg["vis"]["save_dir"]
        args["vis_steps"]         = cfg["vis"]["vis_steps"]
        args["vis_points"]        = cfg["vis"]["vis_points"]
        args["feature_extractor"] = feature_extractor
        args["mapping"]           = mapping
        args["real_embeds"]       = real_embeds

    # <================== Train

    losses = train(perturb_net, G, classifier, target, **args)


    # <================== Save figure and checkpoint

    plt.figure(figsize=(10,6), dpi=256)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(cfg['log_dir'], 'loss.png'))

    torch.save(perturb_net, 'output/model.pt')


    perturb_net.eval()
    with torch.no_grad():
        images = generate_samples(100, cfg["latent_dim"], G, perturb=perturb_net, device=cfg["device"])

    classes = classifier(images)
    class_nums = classes.argmax(-1)

    plt.figure(figsize=(5, 5), dpi=256)
    plt.hist(class_nums.detach().cpu().numpy())
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(cfg['log_dir'], 'hist.png'))


if __name__ == '__main__':

    cfg = load_default_config()
    print(cfg)

    run_experiment(cfg)