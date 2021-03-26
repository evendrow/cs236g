import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
# import seaborn as sns

import umap
import umap.plot

from src.utils import gen_noise, gen_images

def get_umap_embedding(images, feature_extractor):
    feats = feature_extractor.get_feats(images).cpu().numpy()
    mapper = umap.UMAP().fit(feats)
    embeds = mapper.transform(feats)
    return (mapper, embeds)


# generate samples from gan
def generate_samples(sample_size, latent_dim, G, perturb=None, batch_size=64, device='cuda'):
    images_list = []
    with torch.no_grad():
        for i in range(sample_size//batch_size):
            noise = gen_noise(sample_size//10, latent_dim, device=device)
            _, images, _, _ = gen_images(noise, G, None, None, perturb=perturb)
            images_list.append(images)
    images_list = torch.cat(images_list)

    # perturbed_feats = feature_extractor.get_feats(images_perturbed).cpu().numpy()

    return images_list


def umap_plot_images(embs, labels, images, xlim=None, ylim=None, cmap='bwr', show_legend=False):
    fig = plt.figure(figsize=(38,19), dpi=32, constrained_layout=False)

    grid_size=5

    # fig.tight_layout()
    gs = fig.add_gridspec(grid_size, grid_size*2, wspace=0, hspace=0)
    umap_ax = fig.add_subplot(gs[:, :grid_size])
    im_axes = []
    for i in range(grid_size):
        for j in range(grid_size):
            im_axes.append(fig.add_subplot(gs[i, j+grid_size]))
            im_axes[-1].set_axis_off()
    #         im_axes[-1].set(xticks=[], yticks=[])

    for i in range(grid_size**2):
        im_axes[i].imshow(images[i][0], aspect='auto', cmap='gray')

    scatter = umap_ax.scatter(embs[:,0], embs[:,1], s = 1, c = labels, cmap=cmap)
    if show_legend:
        # produce a legend with the unique colors from the scatter
        legend1 = umap_ax.legend(*scatter.legend_elements(),
                            title="", fontsize=25, markerscale=4)
        umap_ax.add_artist(legend1)
    umap_ax.get_xaxis().set_ticks([])
    umap_ax.get_yaxis().set_ticks([])
    umap_ax.set_axis_off()
    if xlim is not None and ylim is not None:
        umap_ax.set_xlim(xlim)
        umap_ax.set_ylim(ylim)
    
    return fig

def plot_two_types(emb1, emb2, images):
    all_embs_labels = np.concatenate((np.zeros(emb1.shape[0]), np.ones(emb2.shape[0])))
    all_embs = np.concatenate((emb1, emb2))
    xlim = [min(emb1[:,0])*1.1, max(emb1[:,0])*1.1]
    ylim = [min(emb1[:,1])*1.1, max(emb1[:,1])*1.1]
    return umap_plot_images(all_embs, all_embs_labels, images, xlim=xlim, ylim=ylim) #cmap='Spectral', show_legend=True)


