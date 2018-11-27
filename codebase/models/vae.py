import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################

        #sample z from encoder distribution
        q_m, q_v = self.enc.encode(x)
        #print("q_m", q_m.size())
        z_given_x = ut.sample_gaussian(q_m, q_v)
        decoded_bernoulli_logits = self.dec.decode(z_given_x)
        rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits)
        rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.shape), torch.ones(q_v.shape)

        kl = ut.kl_normal(q_m, q_v, p_m, p_v)
        kl = torch.mean(kl)

        nelbo = rec + kl

        #kl = ut.kl_normal(q_m, q_v, p_m, p_v)
        #print("kl_size", kl.size())

        #nelbo = (-rec + kl)*torch.tensor(1/x.size(0))

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
        
        q_m, q_v = self.enc.encode(x)
        q_m_, q_v_ = ut.duplicate(q_m, rep=iw), ut.duplicate(q_v, rep=iw)

        z_given_x = ut.sample_gaussian(q_m_, q_v_)
        decoded_bernoulli_logits = self.dec.decode(z_given_x)

        #duplicate x
        x_dup = ut.duplicate(x, rep = iw)
        
        rec = ut.log_bernoulli_with_logits(x_dup, decoded_bernoulli_logits)
    

        #compute kl
        p_m, p_v = torch.zeros(q_m.shape), torch.ones(q_v.shape)
        p_m_, p_v_ = ut.duplicate(p_m, iw), ut.duplicate(p_v, iw)
        #print("p_m", p_m.shape)
        log_q_phi = ut.log_normal(z_given_x, q_m_, q_v_) #encoded distribution
        log_p = ut.log_normal(z_given_x, p_m_, p_v_) #prior distribution

        kl = log_q_phi - log_p

        niwae = rec - kl
        
        #reshape to size (iw, bs) and then sum
        niwae = ut.log_mean_exp(niwae.reshape(iw, -1), dim=0)
        rec = ut.log_mean_exp(rec, dim = 0)
        kl = ut.log_mean_exp(kl, dim = 0)

        niwae = -torch.mean(niwae)
        kl = torch.mean(kl)
        rec = torch.mean(kl)
        
        

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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
