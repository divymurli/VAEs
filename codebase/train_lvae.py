import argparse
import numpy as np
import os
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

def train(model, train_loader, device, tqdm,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
	
	# Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    beta = ut.DeterministicWarmup(n=iter_max, t_max = 1)


