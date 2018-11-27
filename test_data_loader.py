import torch
from torchvision import datasets, transforms
import numpy as np
import os,sys
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.models.lvae import LVAE
from codebase.models.lvae import DeterministicWarmup
from codebase.models.nns import v1
from codebase.models.nns import v3


preprocess = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, download=True, transform=preprocess),
		batch_size=50, shuffle=True)

x, y = next(iter(train_loader))
print(x.shape)
#print(y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.bernoulli(x.to(device).reshape(x.size(0), -1))
y = y.new(np.eye(10)[y]).to(device).float()
print(x.shape)
#print(y)

enc = v1.Encoder(5)
MLP = v3.MLP(784, 512, 64)
#q_m, q_v = enc.encode(x)
#mean = mean_layer.forward(l_enc_a)
#var  = var_layer.forward(l_enc_a)
l_enc_a, mu, var = MLP.encode(x)
print(l_enc_a.shape)
print(mu.shape)
print(var.shape)

Dec = v3.FinalDecoder()
out = Dec.decode(mu)
print(out.shape)

#test LVAE
lvae = LVAE()
vae = VAE(z_dim=2)
z_down, mu0, var0, _, _ = lvae.EncoderPartialDecoder(x)
print("mu0", mu0.shape)

decoded_logits = lvae.DecodeRest(mu0, var0)
print(decoded_logits.shape)

nelbo, _, _ = lvae.negative_elbo_bound(x,1)
print(nelbo)
loss, summaries = lvae.loss(x,1)


train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
beta = DeterministicWarmup(n=50, t_max=1)

for i in range(10):
	print(next(beta))

	
#train_counter = list(enumerate(train_loader,1))
#print(len(train_counter))
#print(mean.shape)
#print(var.shape)
#var = var.reshape((-1,1,1,64))
#print(var.shape)
#var = var.reshape((-1, 64))
#print(var.shape)
#out = enc.encode_2(x)
#print(out.shape)

#print(q_m.size())