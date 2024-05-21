'''
Pytorch version of PIMA 
'''

import sys
import os
import random
import numpy as np

from lib.evaluate import *
from lib.plotting import * 


import argparse
import torch
torch.use_deterministic_algorithms(True)
from torch.nn import functional as F
import yaml
import json
from types import SimpleNamespace

from lib.utils import get_device
from lib.train_utils import *
from lib.data_utils import *

#Training loop
def train(opt, cfg, experimentID, ckpt_path, train_dataloader, val_dataloader, test_dataloader, model, unimodal_model, optimizer):
    device = get_device(opt.gpu)
    training_model = unimodal_model if "Unimodal" in opt.model else model
    #Training
    best_loss = 1e9
    for itr in range(1, opt.num_epochs + 1):
        # Making sure model is in training mode
        if not "Unimodal" in opt.model:
            model.train()
        else: #unimodal uses final mulitmodal weights for training
            model.eval()
            unimodal_model.train()
        model.unimodal = False #do not run encoder, use pretrained model encodingsmodel.unimodal = False

        if not "Unimodal" in opt.model and opt.EM:
            EM_step(opt,model, train_dataloader)

        train_loss = 0

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            XS = [x.to(device) for x in batch["inputs"]]
            gamma, pi, mu_q, logvar_q, decoders_out = model(XS)
            loss, target_loss = calc_loss(opt, XS, gamma, pi, mu_q, logvar_q, decoders_out, training_model, mode="train", itr=itr, l1_reduction="sum")
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader.dataset)
        if opt.use_wandb:
            wandb.log({"loss/train": train_loss})
        if opt.tensorboard:
            opt.writer.add_scalar("loss/train", train_loss, itr)
        message = 'Epoch {:04d} | Loss {:.6e} |'.format(itr, train_loss)
        
        #VIZ
        if itr%opt.viz_frequency==0:
            viz_epoch(opt, train_dataloader, model, itr, unimodal_model=unimodal_model)

        # Val
        model.eval()
        if not unimodal_model is None:
            unimodal_model.eval()
        model.unimodal = False #do not run encoder, use pretrained model encodings

        val_loss = 0
        target_loss = 0
        for i, batch in enumerate(val_dataloader):
            XS = [x.to(device) for x in batch["inputs"]]
            gamma, pi, mu_q, logvar_q, decoders_out = model(XS)
            loss, L_target = calc_loss(opt, XS, gamma, pi, mu_q, logvar_q, decoders_out, training_model, mode="val", itr=itr, l1_reduction="sum")
            val_loss += loss.item()
            target_loss += L_target

        if opt.use_wandb and not opt.classification:
            wandb.log({"loss/target": target_loss})
        val_loss /= len(val_dataloader.dataset)
        if opt.use_wandb:
            wandb.log({"loss/val": val_loss})
        if opt.tensorboard:
            opt.writer.add_scalar("loss/val", val_loss, itr)
        message = 'Epoch {:04d} | Val Loss {:.6e} |'.format(itr, val_loss)
        print(message)
        best_loss = checkpoint_model(opt, best_loss, itr, training_model, optimizer, val_loss, target_loss, ckpt_path)
        #VIZ
        if itr%opt.viz_frequency==0:
            viz_epoch(opt, val_dataloader, model, itr, unimodal_model=unimodal_model, mode="val")

    #Write best model to wandb as Artifact
    if opt.use_wandb:
        write_model_artifact(opt,ckpt_path)

    #Evaluate best model
    with torch.no_grad():
        if not "Unimodal" in opt.model:
            print("Loading saved model {}".format(ckpt_path))
            model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        else:
            unimodal_model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

        res = infer(opt, train_dataloader, model, unimodal_model=unimodal_model, mode="train")
        log_metrics(opt, res, mode="train", tag="_final", itr=itr)
        res = infer(opt, val_dataloader, model, unimodal_model=unimodal_model, mode="val")
        log_metrics(opt, res, mode="val", tag="_final", itr=itr)
        res = infer(opt, test_dataloader, model, unimodal_model=unimodal_model, mode="test")
        log_metrics(opt, res, mode="test", tag="_final", itr=itr)
    if opt.use_wandb:
        wandb.finish()

def main(opt):
    cfg, experimentID, ckpt_path = init_run(opt)
    print("Experiment " + str(experimentID))
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(opt)
    model, unimodal_model, optimizer = init_models(opt, cfg, train_dataloader.dataset)
    train(opt, cfg, experimentID, ckpt_path, train_dataloader, val_dataloader, test_dataloader, model, unimodal_model, optimizer)
    if opt.tensorboard:
        opt.writer.close()
    
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="yaml configuration file")
    parser.add_argument("--gpu", type=int, help="(int) gpu number", nargs="?")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    if args.gpu:
        cfg_dict["gpu"] = int(gpu)
    opt = json.loads(json.dumps(cfg_dict), object_hook=lambda d: SimpleNamespace(**d))
    main(opt)
    
if __name__=='__main__':
    run()
