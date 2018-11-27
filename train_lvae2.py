import torch
from codebase import utils as ut
from codebase.models.lvae import LVAE
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dim',      type=int, default=784,   help="Number of input dimensions")
parser.add_argument('--layer1_dim',  type=int, default=512,   help="Number of first layer dimensions")
parser.add_argument('--layer2_dim',  type=int, default=256,   help="Number of second layer dimensions")
parser.add_argument('--latent1_dim', type=int, default=64,    help="Number of first layer latent dimensions")
parser.add_argument('--latent2_dim', type=int, default=32,    help="Number of second layer dimensions")
parser.add_argument('--epoch_max',   type=int, default=50,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
args = parser.parse_args()

layout = [
	('model={:s}',  'lvae'),
	('layer1_dim={:02d}',  args.layer1_dim),
	('layer2_dim={:03d}', args.layer2_dim),
	('latent1_dim={:04d}', args.latent1_dim),
	('latent2_dim={:04d}', args.latent2_dim),
	('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[Sønderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t
lvae = LVAE(name=model_name, in_dim=args.in_dim, layer1_dim=args.layer1_dim, layer2_dim=args.layer2_dim, 
	latent1_dim=args.latent1_dim, latent2_dim=args.latent2_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=50, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch



for epoch in range(args.epoch_max):
	lvae.train()
	total_loss = 0
	total_rec = 0
	total_kl = 0
	for (u, _) in train_loader:
		#u = Variable(u)
		#print(u.shape)
		#if cuda: u = u.cuda(device=0)
		optimizer.zero_grad()
		u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
		
		L, kl, rec = lvae.negative_elbo_bound(u, next(beta))
		#print(model.kl_divergence)
		

		L.backward()
		optimizer.step()
		#optimizer.zero_grad()
		#print(L)
		total_loss += L.data[0]
		total_kl += kl.data[0]
		total_rec += rec.data[0]

	m = len(train_loader)

	if epoch % 1 == 0:
		print(f"Epoch: {epoch+1}\tL: {total_loss/m:.2f}\tkl: {total_kl/m:.2f}\trec: {total_rec/m:.2f}")
	if epoch % args.iter_save == 0:
		ut.save_model_by_name(lvae, epoch)
