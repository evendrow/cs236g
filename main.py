import torch

from src.train import train
from src.models import PerturbationNetwork
from src.mnist import load_mnist_gan, load_mnist_classifier, MNISTFeatureExtractor

def run_experiment(cfg):

    args = dict()
    args["batch_size"] = cfg["batch_size"]
    args["device"] = cfg["device"]
    args["latent_dim"] = cfg["latent_dim"]
    args["class_weight"] = cfg["class_weight"]
    args["perturb_weight"] = cfg["perturb_weight"]
    args["disc_weight"] = cfg["disc_weight"]
    args["iters"] = cfg["iters"]
    args["log_steps"] = cfg["log_steps"]

    perturb_net = PerturbationNetwork(latent_dim=cfg["latent_dim"], 
                                      num_hidden_layers=cfg["perturb_cfg"]["hidden_layers"],
                                      dropout=cfg["perturb_cfg"]["dropout"])
    perturb_net.to(device=cfg["device"])

    G, D = load_mnist_gan(device=cfg["device"])
    classifier = load_mnist_classifier(device=cfg["device"])
    feature_extractor = MNISTFeatureExtractor(device=cfg["device"])

    target = torch.Tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(cfg["device"])

    train(perturb_net, G, classifier, target, **args)

if __name__ == '__main__':

    cfg = dict()
    cfg["batch_size"] = 8
    cfg["device"] = "cpu"
    cfg["latent_dim"] = 100
    cfg["class_weight"] = 100
    cfg["perturb_weight"] = 100
    cfg["disc_weight"] = 100
    cfg["iters"] = 300
    cfg["log_steps"] = 10

    cfg["perturb_cfg"] = dict()
    cfg["perturb_cfg"]["hidden_layers"] = 4
    cfg["perturb_cfg"]["dropout"] = 0.1

    run_experiment(cfg)