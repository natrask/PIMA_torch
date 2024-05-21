import torch
from lib.encoder_decoder import *
from lib.VAE import *
from torch.nn import ModuleList
import numpy as np
from lib.expert_models import *

def create_GeneralPIMA(opt, device, dataset=None):
    print('Creating a General PIMA model')
    latent_dim = opt.encoding_dim
    input_dim = opt.img_size
    num_clusters = opt.num_clusters
    encoders = ModuleList()
    decoders = ModuleList()

    #If config doesn't supply data channel info; default to 1 channel
    if not hasattr(opt, "channels_in_1d"):
        opt.channels_in_1d = 1
    if not  hasattr(opt, "channels_in_2d"):
        opt.channels_in_2d = 1

    #Create PIMA with lattice expert model
    if opt.dataset == "lattice" and opt.data_driven==False:
        encoders.append(EncoderConv2D(input_dim, latent_dim, in_channels=opt.channels_in_2d))
        encoders.append(EncoderConv1D(opt.num_1d_sample_points, latent_dim, in_channels=opt.channels_in_1d))
        decoders.append(DecoderConv2D(latent_dim, input_dim, in_channels=opt.channels_in_2d))
        decoders.append(StressStrainExpert(opt.num_1d_sample_points, num_clusters, device))
    elif opt.dataset == "mnist" and opt.data_driven==False:
        encoders.append(EncoderConv2D(input_dim, latent_dim, in_channels=opt.channels_in_2d))
        encoders.append(EncoderConv1D(opt.num_1d_sample_points, latent_dim, in_channels=opt.channels_in_1d))
        decoders.append(DecoderConv2D(latent_dim, input_dim, in_channels=opt.channels_in_2d))
        decoders.append(MNISTExpert(opt.num_1d_sample_points, num_clusters, device))
    else: #data driven
        XS = dataset[0]["inputs"]
        for X in XS:
            #Assume shape determines correct encoder/decoder selection
            if len(X.shape) == 1: #parameter set
                encoders.append(EncoderSet(opt.n_params, latent_dim))
                decoders.append(DecoderMLP1D(opt,latent_dim, opt.n_params))
            elif len(X.shape) == 2: #1D data
                encoders.append(EncoderConv1D(opt.num_1d_sample_points, latent_dim, in_channels=opt.channels_in_1d))
                decoders.append(DecoderMLP1D(opt,latent_dim, opt.num_1d_sample_points,in_channels=opt.channels_in_1d))
            else: #1 channel 2D data
                encoders.append(EncoderConv2D(input_dim, latent_dim, in_channels=opt.channels_in_2d))
                decoders.append(DecoderConv2D(latent_dim, input_dim, in_channels=opt.channels_in_2d))

    model = GeneralPIMA(opt,encoders, decoders)
    return model


def create_unimodal_PIMA(opt, device):
    print('Creating a Unimodal PIMA model')
    if not hasattr(opt, "channels_in_1d"):
        opt.channels_in_1d = 1
    if not  hasattr(opt, "channels_in_2d"):
        opt.channels_in_2d = 1

    latent_dim = opt.encoding_dim
    input_dim = opt.img_size
    num_clusters = opt.num_clusters
    if opt.modality_in == 0:
        encoder = EncoderConv2D(input_dim, latent_dim, in_channels=opt.channels_in_2d)
    else: #modality_in == 1
        encoder = EncoderConv1D(opt.num_1d_sample_points, latent_dim, in_channels=opt.channels_in_1d)
    if opt.modality_out == 0:
        decoder = DecoderConv2D(latent_dim, input_dim, in_channels=opt.channels_in_2d)
    else:
        if opt.dataset == "lattice":
            decoder = StressStrainExpert(opt.num_1d_sample_points, num_clusters, device)
            data_driven = False
        elif opt.dataset == "mnist":
            decoder = MNISTExpert(opt.num_1d_sample_points, num_clusters, device)
            data_driven = False
        else: #data driven
            decoder = DecoderMLP1D(opt,latent_dim, opt.num_1d_sample_points,in_channels=opt.channels_in_1d)
            data_driven = True
    
    model = UnimodalPIMA(input_dim=input_dim,
                             latent_dim=latent_dim,
                             encoder=encoder,
                             decoder=decoder,
                             device=device,
                             Ndata=opt.num_1d_sample_points, 
                             dataset=opt.dataset,
                             unimodal_in = opt.modality_in,
                             unimodal_out = opt.modality_out,
                             data_driven=data_driven)
    return model
