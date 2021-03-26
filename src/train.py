import torch
import torch.nn as nn
from src.loss import DistributionLoss

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


def train(perturb_net, G, classifier, target, D=None, batch_size=64, latent_dim=100, class_weight=1, 
          perturb_weight=1, disc_weight=1, iters=300, save_dir=None, feature_extractor=None, 
          mapping=None, real_embeds=None, log_steps=50, device='cuda'):
    """
    Trains latent perturbation network with provided arguments

    Arguments:
        perturb_net    -- latent perturbation network
        G              -- generator network
        classifier     -- classification network
        target         -- target distribution (for distribution loss)
        D              -- discriminator network
        batch size     -- size of batches used for training
        latent_dim     -- latent dimension of generator
        class_weight   -- weight given to classification loss (distribution/fairness loss)
        perturb_weight -- weight given to perturbation loss
        disc_weight    -- weight given to distriminator loss
        iters          -- number of training iterations
        log_steps      -- frequency of loss logging
        device         -- device on which to perform operations

        (Visualization)
        save_dir       -- directory to save visualization frames
        feat_extractor -- feature extractor network
        mapping        -- UMAP mapping object for transforming image features
        real_embeds    -- UMAP embeddings of real images

    Returns:
        losses -- list of losses for each training iteration
    """

    print("TRAIN")
    
    if not (save_dir is None and mapping is None and real_embeds is None and feature_extractor is None):
        if not (save_dir is not None and mapping is not None and real_embeds is not None and feature_extractor is not None):
            print("Need all arguments if saving frames, but not all provided")
            return
    
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    optimizer = torch.optim.Adam(perturb_net.parameters())
    
    class_loss_fn = DistributionLoss() #nn.CrossEntropyLoss()
    perturb_loss_fn = nn.MSELoss()
    disc_loss_fn = nn.BCELoss()
    
    # noise for visualiztion
    vis_noise = gen_noise(25, latent_dim, device=device)

    losses = []
    for i in range(iters):

        # Train one iteration
        optimizer.zero_grad()
        
        
        noise = gen_noise(batch_size, latent_dim, device=device)
            
        noise_p, images, preds, disc_score = gen_images(noise, G, D, classifier, perturb=perturb_net)
        
        disc_target = torch.ones(batch_size).to(device) # 1 = real
         

        class_loss = class_loss_fn(preds, target)
        perturb_loss = perturb_loss_fn(noise_p, noise)
        loss = (class_weight * class_loss) + (perturb_weight * perturb_loss)

        # if D is not None:
            # disc_loss = disc_loss_fn(disc_score, disc_target)
            # loss += (disc_weight * disc_loss)

        loss.backward()
        optimizer.step()
        
        # Visualization
        
        if save_dir is not None and i % 3 == 0:
            with torch.no_grad():
                
                feat_noise = gen_noise(3072, latent_dim, device=device)
                _, images, _, _ = gen_images(feat_noise, G, D, classifier, perturb_net)
                image_feats = feature_extractor.get_feats(images).cpu().numpy()
                new_embeds = mapping.transform(image_feats)
                
                _, images, _, _ = gen_images(vis_noise, G, D, classifier, perturb_net)
                fig = plot_two_types(real_embeds, new_embeds, images.detach().cpu().numpy())
                
                filename = os.path.join(save_dir, 'frame'+str(i)+'.png')
                fig.savefig(filename, dpi=72)
                plt.close(fig)

        losses.append(loss.item())
        if i % log_steps == 0:
            print('Iter', i, '-- Loss', loss.item())
            
    return losses
