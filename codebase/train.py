import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

def train(model, train_loader, labeled_subset, device, tqdm, writer,
		  iter_max=np.inf, iter_save=np.inf,
		  model_name='model', y_status='none', reinitialize=False):
	# Optimization
	if reinitialize:
		model.apply(ut.reset_weights)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	i = 0
	#beta = ut.DeterministicWarmup(n=50, t_max = 1)
	beta = 0
	with tqdm(total=iter_max) as pbar:
		while True:
			for batch_idx, (xu, yu) in enumerate(train_loader):
				i += 1 # i is num of gradient steps taken by end of loop iteration
				optimizer.zero_grad()
				if i%600==0:
					beta += 0.02
				#print(beta)
				if y_status == 'none':
					xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
					yu = yu.new(np.eye(10)[yu]).to(device).float()
					loss, summaries = model.loss(xu, beta)
					#print(beta)
					#print(next(beta))
					#loss, summaries = model.loss(xu)
				elif y_status == 'semisup':
					xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
					yu = yu.new(np.eye(10)[yu]).to(device).float()
					# xl and yl already preprocessed
					xl, yl = labeled_subset
					xl = torch.bernoulli(xl)
					loss, summaries = model.loss(xu, xl, yl)

					# Add training accuracy computation
					pred = model.cls.classify(xu).argmax(1)
					true = yu.argmax(1)
					acc = (pred == true).float().mean()
					summaries['class/acc'] = acc

				elif y_status == 'fullsup':
					# Janky code: fullsup is only for SVHN
					# xu is not bernoulli for SVHN
					xu = xu.to(device).reshape(xu.size(0), -1)
					yu = yu.new(np.eye(10)[yu]).to(device).float()
					loss, summaries = model.loss(xu, yu)

				loss.backward()
				optimizer.step()

				# Feel free to modify the progress bar
				if y_status == 'none':
					pbar.set_postfix(
						loss='{:.2e}'.format(loss),
						kl='{:.2f}'.format(summaries['gen/kl_z']),
						rec='{:.2g}'.format(summaries['gen/rec']))
				elif y_status == 'semisup':
					pbar.set_postfix(
						loss='{:.2e}'.format(loss),
						acc='{:.2e}'.format(acc))
				elif y_status == 'fullsup':
					pbar.set_postfix(
						loss='{:.2e}'.format(loss),
						kl='{:.2e}'.format(summaries['gen/kl_z']))
				pbar.update(1)

				# Log summaries
				if i % 50 == 0: ut.log_summaries(writer, summaries, i)

				# Save model
				if i % iter_save == 0:
					ut.save_model_by_name(model, i)

				if i == iter_max:
					return
