'''
Pytorch version of PIMA 
'''

import sys
import os
import random
import numpy as np
from lib.VAE import reparameterize
from lib.evaluate import * 
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.create_models import *
from matplotlib import pyplot as plt
import wandb
import argparse
import torch
from torch.nn import functional as F
import yaml
import json
from types import SimpleNamespace
from pytorch_datasets.bfp_datasets import *
from lib.utils import get_device
from lib.train_utils import init_run, init_models, calc_loss, viz_epoch, infer
from lib.data_utils import init_dataloaders, denormalize

def main_infer(opt):
    cfg, experimentID, ckpt_path = init_run(opt)
    unimodal_model = None
    if opt.use_wandb and opt.use_model_artifact:
        model_artifact = wandb.use_artifact(opt.use_model_artifact)
        artifact_dir = model_artifact.download()
        fs = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
        state_dict = torch.load(os.path.join(artifact_dir,fs[0]))['model_state_dict']
    else:
        state_dict = torch.load(opt.trained_multimodal_model_path)['model_state_dict']



    if "Unimodal" in opt.model:
      if opt.use_wandb and opt.use_unimodal_model_artifact:
          model_artifact = wandb.use_artifact(opt.use_unimodal_model_artifact)
          artifact_dir = model_artifact.download()
          fs = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
          unimodal_state_dict = torch.load(os.path.join(artifact_dir,fs[0]))['model_state_dict']
      else:
          unimodal_state_dict = torch.load(opt.trained_unimodal_model_path)['model_state_dict']


    print("Experiment " + str(experimentID))
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(opt)
    model, unimodal_model, optimizer = init_models(opt, cfg, train_dataloader.dataset)
    model.load_state_dict(state_dict)
    if "Unimodal" in opt.model:
        unimodal_model.load_state_dict(unimodal_state_dict)

    res = infer(opt, train_dataloader, model, unimodal_model=unimodal_model, mode="train")
    viz_epoch(opt, train_dataloader, model, "final", unimodal_model=unimodal_model)
    res = infer(opt, val_dataloader, model, unimodal_model=unimodal_model, mode="val")
    viz_epoch(opt, val_dataloader, model, "final", unimodal_model=unimodal_model, mode="val")
    res =infer(opt, test_dataloader, model, unimodal_model=unimodal_model, mode="test")
    viz_epoch(opt, test_dataloader, model, "final", unimodal_model=unimodal_model, mode="test")

    wandb.finish()
    
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="yaml configuration file")
    parser.add_argument("--gpu", type=int, help="(int)gpu number", nargs="?")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    if args.gpu:
        cfg_dict["gpu"] = int(gpu)
    opt = json.loads(json.dumps(cfg_dict), object_hook=lambda d: SimpleNamespace(**d))
    main_infer(opt)

if __name__=='__main__':
    run()
