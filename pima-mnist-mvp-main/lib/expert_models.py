import torch
import torch.nn as nn
import math
import numpy as np

class StressStrainExpert(nn.Module):
    def __init__(self, Ndata, num_clusters, device):
        super(StressStrainExpert, self).__init__()
        #Expert models 
        #t = tf.constant(x,dtype=tf.float32)
        theta_init = 1
        self.logvar_X2 = nn.Parameter(torch.zeros((1,num_clusters), dtype=torch.float64))
        self.theta = nn.Parameter(torch.Tensor(np.random.normal(theta_init,0.01,num_clusters)))
        self.x0 = nn.Parameter(torch.Tensor(np.random.normal(0.5,0.01,num_clusters)))
        self.y0 = nn.Parameter(torch.Tensor(np.random.normal(0.5*theta_init,0.01,num_clusters)))
        self.device = device
        self.Ndata = Ndata
        self.num_clusters = num_clusters
        t = torch.Tensor(np.linspace(0,1,self.Ndata)).unsqueeze(0)
        self.register_buffer('t', t)
        
    def forward(self,z):
        mask = 1.0*(self.t<self.x0.unsqueeze(1))
        mu_piece1 = (self.y0/self.x0).unsqueeze(1)*self.t
        mu_piece2 = self.y0.unsqueeze(1) + self.theta.unsqueeze(1)*(self.t - self.x0.unsqueeze(1))
        mu_X2 = mask*mu_piece1 + (1.-mask)*mu_piece2
        #logvar_X2 = torch.zeros((1,self.num_clusters)).to(self.device)

        return mu_X2, self.logvar_X2

class MNISTExpert(nn.Module):
    def __init__(self, Ndata, num_clusters, device, trainable_logvar=True):
        super(MNISTExpert, self).__init__()
        if trainable_logvar:
            self.logvar_X2 = nn.Parameter(torch.zeros((num_clusters), dtype=torch.float64))
        else:
            logvar_X2 = torch.zeros(1,num_clusters)
            self.register_buffer('logvar_X2', logvar_X2)

        #Expert models 
        theta_init = 4.5
        self.theta = nn.Parameter(torch.Tensor(np.random.normal(theta_init,0.1,num_clusters)))
        self.device = device
        self.Ndata = Ndata
        t = torch.Tensor(np.linspace(0,1,self.Ndata)).unsqueeze(0)
        self.register_buffer('t', t)
        
    def forward(self,z):
        lines = (self.theta).unsqueeze(1)*self.t 
        return lines, self.logvar_X2
