# Variational Autoencoders and Ladder Variational Autoencoders

This repo contains implementations of a basic variational autoencoder (VAE), a gaussian mixture VAE (GMVAE) and a ladder VAE (LVAE), all implemented in PyTorch 0.4. The former two were done as part of a homework assignment for CS 236 at Stanford University, and the last one is an extension of this starter code to ladder variational autoencoders described in https://arxiv.org/abs/1602.02282. The LVAE implemented here is a two-layer latent model with latent layers of dimensions defaulted at 64 and 32 respectively, and the model outputs (trained on MNIST) after one, six and sixteen epochs can be found in `lvae_outputs/`.

1. To train the LVAE, run `python train_lvae2.py --epoch_max (num_epochs)`. 
1. To train the VAE, run `python run_vae.py`.
1. To train the GMVAE, run `python run_gmvae.py`.

Jupyter notebooks have been provided with sample visualisations for VAE, GMVAE and LVAE (for the last one, see `lvae_outputs/` as stated above). The models to train the LVAE are in `codebase/models/nns/v3.py`, where a two-layer multilayer perceptron is implemented as in https://github.com/casperkaae/LVAE/blob/master/run_models.py. The warmup training scheme is used, and the implementation is taken from https://github.com/wohlert/semi-supervised-pytorch/blob/master/semi-supervised/inference/variational.py.

---

### Dependencies

This code was built and tested using the following libraries

```
tqdm==4.20.0
numpy==1.15.2
torchvision==0.2.1
torch==0.4.1.post2
```
