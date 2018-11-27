import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class LVAE(nn.Module):
	def __init__(self, nn='v3', name='lvae', in_dim=784, layer1_dim=512, layer2_dim=256, latent1_dim=32, latent2_dim=16):
		super().__init__()
		self.name=name
		self.in_dim = in_dim
		self.layer1_dim=layer1_dim
		self.layer2_dim=layer2_dim
		self.latent1_dim=latent1_dim
		self.latent2_dim=latent2_dim

		nn = getattr(nns, nn)
		#initialize required MLPs
		self.MLP1 = nn.MLP(self.in_dim, self.layer1_dim, self.latent1_dim)
		self.MLP2 = nn.MLP(self.layer1_dim, self.layer2_dim, self.latent2_dim)
		self.MLP3 = nn.MLP(self.latent2_dim, self.layer1_dim, self.latent1_dim)
		self.FinalDecoder = nn.FinalDecoder(self.latent1_dim, self.in_dim, self.layer1_dim)

		#self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
		#self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
		#self.z_prior = (self.z_prior_m, self.z_prior_v)

		
	def Encoder(self, x):
		
		#---deterministic upward pass
		#upwards
		l_enc_a0, mu_up0, var_up0 = self.MLP1.encode(x) #first level

		#encoder layer 1 mu, var
		_, qmu1, qvar1 = self.MLP2.encode(l_enc_a0) #second level

		#---stochastic downward pass
		#sample a z on top
		z_down = ut.sample_gaussian(qmu1, qvar1)
		#partially downwards
		_, mu_dn0, var_dn0 = self.MLP3.encode(z_down)
		#compute new mu, sigma at first level as per paper
		prec_up0 = var_up0**(-1)
		prec_dn0 = var_dn0**(-1)

		#encoder layer 0 mu, var
		qmu0 = (mu_up0*prec_up0 + mu_dn0*prec_dn0)/(prec_up0 + prec_dn0)
		qvar0 = (prec_up0 + prec_dn0)**(-1)



		return z_down, qmu0, qvar0, qmu1, qvar1

	def Decoder(self, z_given_x):

		_, pmu0, pvar0 = self.MLP3.encode(z_given_x)

		#last step down, sharing weights with stochastic downward pass from encoder

		z0 = ut.sample_gaussian(pmu0, pvar0)

		#return bernoulli logits
		decoded_logits = self.FinalDecoder.decode(z0)

		return decoded_logits, pmu0, pvar0


	def negative_elbo_bound(self, x, beta):

		z_given_x, qmu0, qvar0, qmu1, qvar1 = self.Encoder(x)
		decoded_bernoulli_logits, pmu0, pvar0 = self.Decoder(z_given_x)

		rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits)
		#print(rec.shape)
		rec = -torch.mean(rec)

		pm, pv = torch.zeros(qmu1.shape), torch.ones(qvar1.shape)
		#print("mu1", mu1)
		kl1 = ut.kl_normal(qmu1, qvar1, pm, pv)
		kl2 = ut.kl_normal(qmu0, qvar0, pmu0, pvar0)
		kl = beta*torch.mean(kl1 + kl2)

		nelbo = rec + kl
		#nelbo = rec
		return nelbo, rec, kl


	def loss(self, x, beta):
		nelbo, rec, kl = self.negative_elbo_bound(x, beta)
		loss = nelbo

		summaries = dict((
			('train/loss', nelbo),
			('gen/elbo', -nelbo),
			('gen/kl_z', kl),
			('gen/rec', rec),
		))

		return loss, summaries

		#Code for sampling

	def sample_sigmoid(self, batch):
		z = self.sample_z(batch)
		return self.compute_sigmoid_given(z)

	def compute_sigmoid_given(self, z):
		logits = self.Decode(z)
		return torch.sigmoid(logits)

	def sample_z(self, batch):
		return ut.sample_gaussian(
			self.z_prior[0].expand(batch, self.z_dim),
			self.z_prior[1].expand(batch, self.z_dim))

	def sample_x(self, batch):
		z = self.sample_z(batch)
		return self.sample_x_given(z)

	def sample_x_given(self, z):
		return torch.bernoulli(self.compute_sigmoid_given(z))


