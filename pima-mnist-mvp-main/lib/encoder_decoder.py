import torch
import torch.nn as nn
from math import floor
import numpy as np
import lib.utils as utils

class EncoderSet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EncoderSet, self).__init__()
        self.fc0 = nn.Linear(input_dim, 100).double()
        self.fc1 = nn.Linear(100, 100).double()
        self.fc2 = nn.Linear(100, 2 * latent_dim).double()
        self.act = nn.ReLU().double()
        self.latent_dim = latent_dim
        self.sigmoid = nn.Sigmoid().double()
        
    def forward(self, data):
        out = self.act(self.fc0(data))
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        mu, logvar = torch.split(out, self.latent_dim, -1)
        return mu, logvar
                                                                                                                                                                                    

class EncoderConv1D(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=1):
        super(EncoderConv1D, self).__init__()
        n_0 = input_dim

        in_channels_layer1 = in_channels
        padding_layer1 = 0
        kernel_layer1 = 3,
        stride_layer1 = 1,
        dilation_layer1 = 1,
        out_channels_layer1 = 8

        padding_layer2 = 0
        kernel_layer2 = 3,
        stride_layer2 = 1,
        dilation_layer2 = 1,
        out_channels_layer2 = 16

        n_1 = floor((n_0 + 2 * padding_layer1 - dilation_layer1[0] * (kernel_layer1[0] - 1) - 1) / stride_layer1[0] + 1)
        n_2 = floor((n_1 + 2 * padding_layer1 - dilation_layer1[0] * (kernel_layer1[0] - 1) - 1) / stride_layer1[0] + 1)

        vec_length = n_2 * out_channels_layer2

        encoder = nn.Sequential(
          nn.Conv1d(in_channels=in_channels_layer1,
                  out_channels=out_channels_layer1,
                kernel_size=kernel_layer1,
             stride=stride_layer1,
             padding=padding_layer1,
          dilation=dilation_layer1
         ),
        nn.ELU(),
        nn.BatchNorm1d(out_channels_layer1),
        nn.Conv1d(in_channels=out_channels_layer1,
                out_channels=out_channels_layer2,
             kernel_size=kernel_layer2,
              stride=stride_layer2,
           padding=padding_layer2,
         dilation=dilation_layer2
         ),
        nn.ELU(),
        nn.BatchNorm1d(out_channels_layer2),
        nn.Flatten(),
        nn.Linear(vec_length, 2 * latent_dim),
        #nn.Sigmoid()    
        ).double()

        self.latent_dim = latent_dim
        self.encoder = encoder
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):
        mu, logvar = torch.split(self.encoder(data), self.latent_dim, -1)
        return mu, logvar
        #return self.sigmoid(mu), self.sigmoid(logvar)

class EncoderConv2D(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=1):
        super(EncoderConv2D, self).__init__()

        nx_0 = input_dim[0]
        ny_0 = input_dim[1]

        padding_layer1 = 0
        kernel_layer1 = (3, 3)
        stride_layer1 = (1, 1)
        dilation_layer1 = (1, 1)
        out_channels_layer1 = 32

        padding_layer2 = 0
        kernel_layer2 = (3, 3)
        stride_layer2 = (1, 1)
        dilation_layer2 = (1, 1)
        out_channels_layer2 = 64

        nx_1 = floor(
            (nx_0 + 2 * padding_layer1 - dilation_layer1[0] * (kernel_layer1[0] - 1) - 1) / stride_layer1[0] + 1)
        ny_1 = floor(
            (ny_0 + 2 * padding_layer1 - dilation_layer1[1] * (kernel_layer1[1] - 1) - 1) / stride_layer1[1] + 1)
        nx_2 = floor(
            (nx_1 + 2 * padding_layer1 - dilation_layer1[0] * (kernel_layer1[0] - 1) - 1) / stride_layer1[0] + 1)
        ny_2 = floor(
            (ny_1 + 2 * padding_layer1 - dilation_layer1[1] * (kernel_layer1[1] - 1) - 1) / stride_layer1[1] + 1)

        vec_length = nx_2 * ny_2 * out_channels_layer2

        encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels_layer1,
                      kernel_size=kernel_layer1,
                      stride=stride_layer1,
                      padding=padding_layer1,
                      dilation=dilation_layer1
                      ),
            #MATCHING TF VN WITH ELU
            #nn.ReLU(),
            nn.ELU(),
            nn.BatchNorm2d(out_channels_layer1),
            nn.Conv2d(in_channels=out_channels_layer1,
                      out_channels=out_channels_layer2,
                      kernel_size=kernel_layer2,
                      stride=stride_layer2,
                      padding=padding_layer2,
                      dilation=dilation_layer2
                      ),
            #nn.ReLU(),
            nn.ELU(),
            nn.BatchNorm2d(out_channels_layer2),
            nn.Flatten(),
            nn.Linear(vec_length, 2*latent_dim),
        ).double()

        self.latent_dim = latent_dim
        self.encoder = encoder

    def forward(self, data):
        mu, logvar = torch.split(self.encoder(data), self.latent_dim, -1)
        return mu, logvar



class DecoderMLP1D(nn.Module):
    def __init__(self, opt, latent_dim, input_dim, in_channels=0):
        super(DecoderMLP1D, self).__init__()
        self.input_dim = input_dim
        self.device = utils.get_device(opt.gpu)
        self.num_clusters = opt.num_clusters
        self.out_channels = in_channels
        if in_channels == 0:
            in_channels = 1 #must have 1 channel of output
        decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, input_dim*in_channels)).double()
    
        self.decoder_mlp = decoder_mlp

    def forward(self, data):
        output = self.decoder_mlp(data)
        #TODO - for data-driven approach, how do we deal with variance?
        if self.out_channels:
            return output.reshape(-1, self.out_channels, self.input_dim)
        else:
            return output


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, input_dim=[32,32]):
        super().__init__()
        self.input_dim = input_dim
    def forward(self, x):
        return x[:, :, :self.input_dim[0], :self.input_dim[1]]


class DecoderConv2D(nn.Module):
    def __init__(self, latent_dim, input_dim, in_channels=1):
        super(DecoderConv2D, self).__init__()
        vec_length = int((input_dim[0]-4)//4 * (input_dim[1]-4)//4 * 32)
        img_square = int(np.sqrt(vec_length / 32))
        decoder = nn.Sequential(
            nn.Linear(latent_dim, vec_length),
            nn.LeakyReLU(0.01),
            Reshape(-1, 32, (input_dim[0]-4)//4, (input_dim[1]-4)//4),
            nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding=0),
            nn.ReLU(),
            #COMMENTING OUT BN TO MATCH TF VN
            #nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=0),
            nn.ReLU(),
            #COMMENTING OUT BN TO MATCH TF VN
            #nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, in_channels, stride=1, kernel_size=3, padding=0),
            Trim(input_dim=input_dim),
            #nn.Sigmoid()
        ).double()

        self.decoder = decoder

    def forward(self, data):
        # print(data.shape)
        output = self.decoder(data)
        return output
