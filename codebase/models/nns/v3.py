import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

#two layer perceptron, hardcoded
class MLP(nn.Module):
	def __init__(self, in_dim=784, out_dim=512, latent_dim=64):
		super(MLP, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.latent_dim = latent_dim
		self.net = nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.LeakyReLU(),
			nn.Linear(out_dim, out_dim),
			nn.LeakyReLU()
		)
		self.mean_layer = nn.Sequential(
			nn.Linear(out_dim, latent_dim)
		)
		self.var_layer = nn.Sequential(
			nn.Linear(out_dim, latent_dim),
			#nn.Softplus()
		)

	def encode(self, x, outmeanvar=False):
		h = self.net(x)
		mu, var = self.mean_layer(h), self.var_layer(h)
		var = F.softplus(var) + 1e-8
		#mu, var = ut.gaussian_parameters(h, dim=1)

		return h, mu, var


class FinalDecoder(nn.Module):
	def __init__(self, in_dim=64, out_dim=784, hidden_dim=512):
		super(FinalDecoder, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_dim = hidden_dim
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, out_dim),
			#nn.Sigmoid()
		)
		
	def decode(self, x):
		h = self.net(x)

		return h




