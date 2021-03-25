import torch
import torch.nn as nn

class FeatureExtractor:
    """ Uses an intermediate layer in an MNIST classifier to
        perform feature extraction, especially for visualization """

    def __init__(self, device='cuda'):
        
        self.device = device

        # We add a hook to an intermediate layer to log its outputs
        self.feat_extractor = load_mnist_classifier(device=device)
        feat_extractor.fc.f6.register_forward_hook(self._hook)

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
    

class PerturbationNetwork(nn.Module):
    """ Latent perturbation network """

    def __init__(self, latent_dim=100, num_hidden_layers=4, dropout=0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = latent_dim*2

        perturb_list = []
        for i in range(num_hidden_layers):
            in_dim = self.latent_dim if i == 0 else self.hidden_size
            perturb_list += [
                nn.Linear(in_dim, self.hidden_size)
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        perturb_list += [nn.Linear(self.hidden_size, self.latent_dim)]

        self._perturbation = nn.Sequential(*perturb_list)


    def forward(self, x):
        perturbation = self._perturbation(x)
        x += perturbation
        return x









