import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from progress.bar import Bar

from src.loss import DistributionLoss
from src.utils import gen_noise, gen_images, AverageMeter
from src.vis import umap_plot_images, plot_two_types

def train(perturb_net, G, classifier, target, logger, D=None, batch_size=64, latent_dim=100, class_weight=1, 
          perturb_weight=1, disc_weight=1, iters=300, save_dir=None, feature_extractor=None, 
          mapping=None, real_embeds=None, vis_steps=3, vis_points=1024, log_steps=50, log_dir='', 
          device='cuda'):
    """
    Trains latent perturbation network with provided arguments

    Arguments:
        perturb_net    -- latent perturbation network
        G              -- generator network
        classifier     -- classification network
        target         -- target distribution (for distribution loss)
        logger         -- logging object

        D              -- discriminator network
        batch size     -- size of batches used for training
        latent_dim     -- latent dimension of generator
        class_weight   -- weight given to classification loss (distribution/fairness loss)
        perturb_weight -- weight given to perturbation loss
        disc_weight    -- weight given to distriminator loss
        iters          -- number of training iterations
        log_steps      -- frequency of loss logging
        log_dir        -- directory for output log
        device         -- device on which to perform operations

        (Visualization)
        save_dir       -- directory to save visualization frames
        feat_extractor -- feature extractor network
        mapping        -- UMAP mapping object for transforming image features
        real_embeds    -- UMAP embeddings of real images
        vis_steps      -- frequency of saving visualization frames
        vis_points     -- number of samples to use for visualization

    Returns:
        losses -- list of losses for each training iteration
    """

    # =============== Visualization sanity checks
    
    if not (save_dir is None and mapping is None and real_embeds is None and feature_extractor is None):
        if not (save_dir is not None and mapping is not None and real_embeds is not None and feature_extractor is not None):
            print("Need all arguments if saving frames, but not all provided")
            return
    
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # =============== Optimizer and Losses

    optimizer = torch.optim.Adam(perturb_net.parameters())
    
    class_loss_fn = DistributionLoss() #nn.CrossEntropyLoss()
    perturb_loss_fn = nn.MSELoss()
    disc_loss_fn = nn.BCELoss()
    
    # Noise for visualiztion
    vis_noise = gen_noise(25, latent_dim, device=device)

    # =============== Logging

    # Training variables
    losses = AverageMeter()
    
    timer = {
        'forw': 0,
        'back': 0,
        'vis': 0
    }

    bar = Bar(f'Training:', fill='#', max=iters)
    start = time.time()
    summary_string = ''

    loss_list = []
    for i in range(iters):

        perturb_net.train()
        optimizer.zero_grad()
        
        # =============== Forward

        noise = gen_noise(batch_size, latent_dim, device=device)
        noise_p, images, preds, disc_score = gen_images(noise, G, D, classifier, perturb=perturb_net)

        timer['forw'] = time.time() - start
        start = time.time()
         
        # =============== Loss

        loss_dict = dict()
        loss_dict['class_l'] = class_weight * class_loss_fn(preds, target)
        loss_dict['perturb_l'] = perturb_weight * perturb_loss_fn(noise_p, noise)
        
        loss = loss_dict['class_l'] + loss_dict['perturb_l']

        if D is not None and disc_weight != 0:
            disc_target = torch.ones(batch_size).to(device) # 1 = real
            loss_dict['disc_l'] = disc_weight * disc_loss_fn(disc_score, disc_target)
            loss = loss + loss_dict['disc_l']

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), noise.size(0))
        loss_list.append(loss.item())

        timer['back'] = time.time() - start
        start = time.time()

        # =============== Visualization

        if save_dir is not None and i % vis_steps == 0:
            perturb_net.eval()

            with torch.no_grad():    
                feat_noise = gen_noise(vis_points, latent_dim, device=device)
                _, images, _, _ = gen_images(feat_noise, G, D, classifier, perturb_net)
                image_feats = feature_extractor.get_feats(images).cpu().numpy()
                new_embeds = mapping.transform(image_feats)
                
                _, images, _, _ = gen_images(vis_noise, G, D, classifier, perturb_net)
                fig = plot_two_types(real_embeds, new_embeds, images.detach().cpu().numpy())
                
                filename = os.path.join(save_dir, 'frame'+str(i)+'.png')
                fig.savefig(filename, dpi=72)
                plt.close(fig)

            timer['vis'] = time.time() - start
            start = time.time()

        # =============== Progress bar and logging


        summary_string = f'({i + 1}/{iters}) | Total: {bar.elapsed_td} | ' \
                         f'ETA: {bar.eta_td:} | loss: {losses.mov_avg:.4f}'

        for k, v in loss_dict.items():
            summary_string += f' | {k}: {v:.2f}'

        for k,v in timer.items():
            summary_string += f' | {k}: {v:.2f}'

        bar.suffix = summary_string
        bar.next()

    bar.finish()
    logger.info(summary_string)
        
    return loss_list
