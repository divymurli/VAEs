import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        q_m, q_v = self.enc.encode(x)
        #print("q_m", q_m.size())
        z_given_x = ut.sample_gaussian(q_m, q_v)
        decoded_bernoulli_logits = self.dec.decode(z_given_x)
        rec = -ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits)
        #rec = -torch.mean(rec)

        #terms for KL divergence
        log_q_phi = ut.log_normal(z_given_x, q_m, q_v)
        #print("log_q_phi", log_q_phi.size())
        log_p_theta = ut.log_normal_mixture(z_given_x, prior[0], prior[1])
        #print("log_p_theta", log_p_theta.size())
        kl = log_q_phi - log_p_theta
        #print("kl", kl.size())

        nelbo = torch.mean(kl + rec)

        rec = torch.mean(rec)
        kl = torch.mean(kl)
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        
        q_m, q_v = self.enc.encode(x)
        q_m_, q_v_ = ut.duplicate(q_m, rep=iw), ut.duplicate(q_v, rep=iw)

        z_given_x = ut.sample_gaussian(q_m_, q_v_)
        decoded_bernoulli_logits = self.dec.decode(z_given_x)

        #duplicate x
        x_dup = ut.duplicate(x, rep = iw)
        
        rec = ut.log_bernoulli_with_logits(x_dup, decoded_bernoulli_logits)

        log_p_theta = ut.log_normal_mixture(z_given_x, prior[0], prior[1])
        log_q_phi = ut.log_normal(z_given_x, q_m_, q_v_)

        kl = log_q_phi - log_p_theta

        niwae = rec - kl

        niwae = ut.log_mean_exp(niwae.reshape(iw, -1), dim=0)
        niwae = -torch.mean(niwae)


        #yay!
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
