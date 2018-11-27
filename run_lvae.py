import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.lvae import LVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dim',      type=int, default=784,     help="Number of input dimensions")
parser.add_argument('--layer1_dim',  type=int, default=512,     help="Number of first layer dimensions")
parser.add_argument('--layer2_dim',  type=int, default=256,     help="Number of second layer dimensions")
parser.add_argument('--latent1_dim', type=int, default=64,     help="Number of first layer latent dimensions")
parser.add_argument('--latent2_dim', type=int, default=32,     help="Number of second layer dimensions")
parser.add_argument('--iter_max',    type=int, default=30000, help="Number of training iterations")
parser.add_argument('--iter_save',   type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
args = parser.parse_args()

layout = [
    ('model={:s}',  'lvae'),
    ('layer1_dim={:02d}',  args.layer1_dim),
    ('layer2_dim={:03d}', args.layer2_dim),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)

lvae = LVAE(name=model_name, in_dim=args.in_dim, layer1_dim=args.layer1_dim, layer2_dim=args.layer2_dim, 
	latent1_dim=args.latent1_dim, latent2_dim=args.latent2_dim)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=lvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    #ut.evaluate_lower_bound(lvae, labeled_subset, run_iwae=args.train == 2)

