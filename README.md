# Latent Perturbation

![header](doc/header.gif)

This repository contains the official implementation for _Controlling GAN Generation with Deep Latent Perturbation_

## Getting Started

Clone the repo:

```
git clone https://github.com/evendrow/cs236g.git
```

Clone this repo to get the GAN models, and place it in the same directory as this repo:
```
git clone https://github.com/csinva/gan-vae-pretrained-pytorch
```

Optionally create a conda environment:
```
conda create -n gans python=3.8
conda activate gans
```

And install requirements:
```
pip install -r requirements.txt
```

Now, you can generate samples from the regular and perturbed MNIST gan by simply running:
```
python run.py
```

## Training

Train the model using the default configuration:
```
python main.py
```

Alternatively, use one of the other provided configs or create your own:
```
python main.py --cfg src/configs/ones.yaml
```

This will create an `experiments` folder in the root directory with logging and visualization outputs generated during training. This will also generate a loss plot, desired histogram, and achieved histogram.
