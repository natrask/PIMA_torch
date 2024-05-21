import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math

import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from lib.utils import get_device

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std, dtype=torch.float64)
    z = mu + eps * std
    return z

class UnimodalPIMA(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder, decoder, device, Ndata, dataset, unimodal_in=0, unimodal_out=1, data_driven=False):
        super(UnimodalPIMA, self).__init__()
        self.data_driven = data_driven
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.Ndata = Ndata
        self.unimodal_in = unimodal_in
        self.unimodal_out = unimodal_out
        
    def forward(self, batch):
        X = batch[self.unimodal_in]
            
        batch_sz = X.shape[0]

        mu_q, logvar_q = self.encoder(X) 

        return mu_q, logvar_q

    def compute_loss(self, mu_pred, logvar_pred, mu_multimodal, logvar_multimodal):
        pred_dist = Normal(mu_pred, torch.sqrt(torch.exp(logvar_pred)))
        multimodal_dist = Normal(mu_multimodal, torch.sqrt(torch.exp(logvar_multimodal)))
        return kl_divergence(pred_dist, multimodal_dist).mean()


class GeneralPIMA(nn.Module):
    def __init__(self, opt, encoders, decoders, unimodal=False):
        super(GeneralPIMA, self).__init__()

        self.classification = opt.classification
        self.data_driven = opt.data_driven
        self.dataset = opt.dataset
        self.input_dim = opt.img_size
        self.latent_dim = opt.encoding_dim
        self.device = get_device(opt.gpu)
        self.unimodal = unimodal
        self.encoders = encoders
        self.decoders = decoders
        self.Ndata = opt.num_1d_sample_points
        self.num_clusters = opt.num_clusters
        self.pi = nn.Parameter(torch.ones(opt.num_clusters, dtype=torch.float64) * (1. / opt.num_clusters))
        self.beta0 = opt.beta0 #Coefficient for target variable in regression tasks
        
        if opt.EM:  # Optimizer does not update this as a param
            self.mu_gmm = nn.Parameter(torch.randn(self.num_clusters, self.latent_dim, dtype=torch.float64), requires_grad=False)
            self.mu_gmm.data.grad = None
        else:
            self.mu_gmm = nn.Parameter(1e-3 * torch.randn(self.num_clusters, self.latent_dim, dtype=torch.float64))

        hpack = np.power(self.num_clusters, -1 / self.latent_dim)
        self.logvar_gmm = nn.Parameter(torch.DoubleTensor(np.log(hpack ** 2) * np.ones((self.num_clusters, self.latent_dim))))

    def forward(self, batch):
        if self.unimodal:#skip encoding
            mu_q, logvar_q = batch
        else:
            XS = batch
            batch_sz = XS[0].shape[0]
            mus = torch.zeros((len(XS), batch_sz, self.latent_dim), dtype=torch.float64).to(self.device)
            logvars  = torch.zeros((len(XS), batch_sz, self.latent_dim), dtype=torch.float64).to(self.device)
            for i in range(len(XS)):
                mus[i], logvars[i] = self.encoders[i](XS[i].double())
            logvar_q = -torch.log((torch.exp(-logvars)).sum(axis=0))
            mu_q =  torch.exp(logvar_q) * ((torch.exp(-logvars) * mus).sum(axis=0))
            
        z = reparameterize(mu_q, logvar_q)

        pi = torch.softmax(self.pi,0)

        px12_c_a = torch.prod(torch.exp(-0.5 * torch.pow(z.unsqueeze(1) -self.mu_gmm, 2) / torch.exp(self.logvar_gmm)),-1)


        px12_c_b = torch.prod(torch.exp(-0.5 * self.logvar_gmm), axis=1)

        px12_c = torch.einsum('bp,p->bp', px12_c_a, px12_c_b)

        gammanum = torch.einsum('p,bp->bp', pi, px12_c) + 1e-10

        gamma = gammanum / torch.sum(gammanum, 1).unsqueeze(-1)

        # for dec in self.decoders:
        #     print(type(dec(z)))
            
        decoders_out = [dec(z) for dec in self.decoders]

        return gamma, pi, mu_q, logvar_q, decoders_out

    def compute_loss(self, batch, gamma, pi, mu_q, logvar_q, decoders_out, l1_reduction="sum"):
        XS = batch
        # reconstruction loss
        L1 = 0
        L_target = torch.zeros(1).to(self.device)
        if not self.data_driven:
            L1 += F.mse_loss(decoders_out[0], XS[0],reduction=l1_reduction)
            L1 = 0.5*L1
            # expert loss
            Qa = 0.5*torch.sum((torch.pow(XS[1]-decoders_out[1][0].unsqueeze(0),2)),-1)*(torch.exp(-decoders_out[1][1]))
            Qb = 0.5*self.Ndata*decoders_out[1][1]
            Q_inter = torch.einsum('bc,bc->b',Qa+Qb,gamma)
            L_expert = self.beta0*torch.sum(Q_inter)

        elif self.classification:
            for i in range(len(XS)):
                L1 += F.mse_loss(decoders_out[i], XS[i],reduction=l1_reduction)
            L1 = 0.5*(L1 + self.beta0*(L_target))
            L_expert = torch.zeros(1).to(self.device)

        else:  #regression with target variable 
            for i in range(len(XS)-1):
                L1 += F.mse_loss(decoders_out[i], XS[i],reduction=l1_reduction)
            L_target = F.mse_loss(decoders_out[-1], XS[-1],reduction=l1_reduction)
            L1 = 0.5*(L1 + self.beta0*(L_target))
            L_expert = torch.zeros(1).to(self.device)

        # KL divergence terms

        L2a = self.logvar_gmm #+ torch.exp(logvar_q).unsqueeze(1)
        L2b = torch.exp(logvar_q).unsqueeze(1) / torch.exp(self.logvar_gmm).unsqueeze(0).double()
        L2c = torch.pow(self.mu_gmm.unsqueeze(0) - mu_q.unsqueeze(1), 2) / torch.exp(self.logvar_gmm).double()
        L2 = 0.5 * torch.sum(torch.einsum('bc,bc->b', gamma.double(), torch.sum(L2a + L2b + L2c, axis=-1).double()))
        L3 = -torch.sum(torch.einsum('bp,bp->b', gamma.double(), torch.log(pi.unsqueeze(0)) - torch.log(gamma.double())))
        L4 = -0.5 * torch.sum(torch.einsum('bz->bz', logvar_q))

        return L1, L_expert, L2, L3, L4, L_target

    def update_mu_gmm(self, new_mu_gmm):
        with torch.no_grad():
            self.mu_gmm.copy_(new_mu_gmm)
            
    def update_logvar_gmm(self, new_logvar_gmm):
        with torch.no_grad():
            self.logvar_gmm.copy_(new_logvar_gmm)
        
