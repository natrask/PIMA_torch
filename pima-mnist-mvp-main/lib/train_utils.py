import os
import random
import torch
import numpy as np
import sys
from lib.utils import *
from lib.evaluate import *
from lib.plotting import *
from lib.create_models import *
import torch.optim as optim

#Configure wandb run
def init_run(opt):
    opt.normalization_parameters = None
    default_cfg = vars(opt)
    os.makedirs(opt.savedir, exist_ok=True)

    experimentID = None
    cfg = None

    #Init W&B run
    if opt.use_wandb:
        import wandb
        wandb.init(project=opt.project, entity=opt.entity, job_type=opt.job_type, config=default_cfg, dir=opt.savedir, settings=wandb.Settings(code_dir="."))
        # Sweep config - overwrite opt with sweep values
        cfg = wandb.config
        for k in vars(opt):
            if k in vars(cfg)["_items"]:
                setattr(opt, k, vars(cfg)["_items"][k])
        opt.num_clusters = cfg["num_clusters"]

        experimentID = wandb.run.name

    #Set default cfg values
    if not hasattr(opt, "model_selection"):
        opt.model_selection = "val_loss"
    if not hasattr(opt, "classification"):
        opt.classification = False
    if not hasattr(opt, "beta0"):
        opt.beta0 = 1
    if not hasattr(opt, "beta1"):
        opt.beta1 = 1
    if not hasattr(opt, "viz_frequency"):
        opt.viz_frequency = 100
    if not hasattr(opt, "tensorboard"):
        opt.tensorboard = False
    if not hasattr(opt, "use_model_artifact"):
        opt.use_model_artifact = False
    if not hasattr(opt, "use_wandb"):
        opt.use_wandb = False
        
    if experimentID is None:
        experimentID = random.randint(0,1e8)
        if opt.use_wandb:
            wandb.run.name = str(experimentID)
    opt.experimentID = experimentID

    #Set seeds for reproducibility
    random.seed(opt.rng_seed)
    torch.manual_seed(opt.rng_seed)
    np.random.seed(opt.rng_seed)
    torch.cuda.manual_seed_all(opt.rng_seed)

    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join("TB","{}_{}".format(opt.name,experimentID)))
        opt.writer = writer
    model_dir = os.path.join(opt.savedir, str(opt.name), str(experimentID), 'models')
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path =  os.path.join(model_dir,'trained_model.ckpt')
    return cfg, experimentID, ckpt_path

#Initialize model(s)
def init_models(opt, cfg, train_dataset):
    device = get_device(opt.gpu)
    optimizer = None
    #Create multimodal model
    if opt.model == 'GeneralPIMA' or opt.model == 'PIMA':
        model = create_GeneralPIMA(opt, device, dataset=train_dataset)
        optimizer = optim.Adam(model.parameters(), lr=opt.lrate)
        unimodal_model = None
        
    elif opt.model == "UnimodalPIMA":
        model = create_GeneralPIMA(opt, device, dataset=train_dataset)
        if opt.use_wandb and opt.use_model_artifact:
            model_artifact = wandb.use_artifact(opt.use_model_artifact)
            artifact_dir = model_artifact.download()
            fs = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
            state_dict = torch.load(os.path.join(artifact_dir,fs[0]))['model_state_dict']
        else:
            state_dict = torch.load(opt.trained_multimodal_model_path)['model_state_dict']
        model.load_state_dict(state_dict)
        unimodal_model = create_unimodal_PIMA(opt, device)
        unimodal_model = unimodal_model.to(device)        
        optimizer = optim.Adam(unimodal_model.parameters(), lr=opt.lrate)

    else:
        print("MODEL {} NOT IMPLEMENTED".format(opt.model))
        sys.exit()
    
    model = model.to(device)
    return model, unimodal_model, optimizer


def log_metrics(opt, res, mode, tag="", itr=0):
    if opt.classification: #supervised classification problem
        print({"acc{}/{}".format(tag,mode),res.item()})
        if opt.use_wandb:
            wandb.log({"acc{}/{}".format(tag,mode):res.item()})
        if opt.tensorboard:
            opt.writer.add_scalar("acc{}/{}".format(tag,mode), res.item(), itr)
    else:
        print({"mse{}/{}".format(tag,mode),res.item()})
        if opt.use_wandb:
            wandb.log({"mse{}/{}".format(tag,mode):res.item()})
            wandb.log({"rmse{}/{}".format(tag,mode):torch.sqrt(res).item()})
        if opt.tensorboard:
            opt.writer.add_scalar("mse{}/{}".format(tag,mode),res.item(), itr)
            opt.writer.add_scalar("rmse{}/{}".format(tag,mode),torch.sqrt(res).item(), itr)
    
#Monitor training progress
def viz_epoch(opt, dataloader, model, itr, unimodal_model, mode="train"):
    viz_batch(opt, dataloader, model, itr, unimodal_model=unimodal_model, mode=mode)
    res = infer(opt, dataloader, model, unimodal_model=unimodal_model, mode=mode)
    log_metrics(opt, res, mode, itr=itr)

#Calculate loss 
def calc_loss(opt, XS, gamma, pi, mu_q, logvar_q, decoders_out, model, mode="train", itr=0, l1_reduction="sum"):
    if not "Unimodal" in opt.model:
        L1, L_expert, L2, L3, L4, L_target = model.compute_loss(XS, gamma, pi, mu_q, logvar_q, decoders_out, l1_reduction=l1_reduction)
        batch_sz =  len(XS[0])
        if opt.use_wandb:
            wandb.log({"L1_{}".format(mode): L1/batch_sz, "L_expert_{}".format(mode): L_expert/batch_sz, "L2_{}".format(mode): L2/batch_sz, "L3_{}".format(mode): L3/batch_sz, "L4_{}".format(mode): L4/batch_sz})
        if opt.tensorboard:
            opt.writer.add_scalar("L1_{}".format(mode),L1/batch_sz, itr)
            opt.writer.add_scalar("L_expert_{}".format(mode), L_expert/batch_sz, itr)
            opt.writer.add_scalar("L2_{}".format(mode), L2/batch_sz, itr)
            opt.writer.add_scalar("L3_{}".format(mode), L3/batch_sz, itr)
            opt.writer.add_scalar("L4_{}".format(mode), L4/batch_sz, itr)            
        loss = L1 + L_expert + opt.beta1*(L2 + L3 + L4)
        target_loss = L_target.item()
    else:
        mu, logvar = model(XS)
        loss = model.compute_loss(mu, logvar, mu_q, logvar_q)
        target_loss = loss.item()
    return loss, target_loss

#After training, store best model in wandb
def write_model_artifact(opt,ckpt_path):
    if "Unimodal" in opt.model:
        model_name = "trained_unimodal_model_mode_{}".format(opt.modality_in)
    else:
        model_name = "trained_multimodal_model"
    model_artifact = wandb.Artifact(model_name, type="model")
    model_artifact.add_file(ckpt_path)
    wandb.log_artifact(model_artifact)

#Save model if metric (usually val_loss) indicates model improvement during training
def checkpoint_model(opt, best_loss, itr, model, optimizer, loss, target_loss, ckpt_path):
    if opt.model_selection == "target":
        if target_loss < best_loss:
            best_loss = target_loss
            print("NEW BEST, saving model {}".format(ckpt_path))
            torch.save({'epoch': itr, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': target_loss}, ckpt_path)
    else:
        if loss < best_loss:
            print("NEW BEST, saving model {}".format(ckpt_path))
            torch.save({'epoch': itr, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, ckpt_path)
            best_loss = loss

    return best_loss


def EM_step(opt, model, dataloader):
    with torch.no_grad():
        device = get_device(opt.gpu)
        wsum = torch.zeros((opt.num_clusters,), dtype=torch.float64)

        mean = torch.zeros((opt.num_clusters, opt.encoding_dim), dtype=torch.float64)

        for i, batch in enumerate(dataloader):
            XS = [x.to(device) for x in batch["inputs"]]

            gamma, pi, mu_q, logvar_q, decoders_out = model(XS)
            gamma = gamma.cpu().double()#.numpy()
            mu_q = mu_q.cpu().double()#.numpy()
            wsumold = wsum
            wsum = wsum + gamma.sum(0)
            mean_old = mean
            mean = (wsumold.unsqueeze(-1) * mean_old + (gamma.unsqueeze(-1) * mu_q.unsqueeze(1)).sum(0)) / wsum.unsqueeze(1)
            model.update_mu_gmm(mean)


#Run inference over dataset
def infer(opt, dataloader, model, unimodal_model=None, mode="train"):
    if not model is None:
        model.eval()
    device = get_device(opt.gpu)
    experimentID = opt.experimentID
    plotdir = os.path.join(opt.savedir,str(opt.name), str(experimentID), "plots")
    os.makedirs(plotdir, exist_ok=True)

    gammas = None
    
    for i, batch in enumerate(dataloader):
        XS = [x.to(device) for x in batch["inputs"]]
        if not opt.classification: 
            labels = batch["regression_lbls"]
        else:
            labels = batch["class_lbls"]
            labels = labels.argmax(axis=-1)
        if unimodal_model is None:
            gamma, pi, mu_q, logvar_q, decoders_out = model(XS)
        else:
            if opt.dataset == "lattice" or opt.dataset == "mnist":
                mu_q, logvar_q = unimodal_model(XS)
                model.unimodal = True
                gamma, pi, mu_q, logvar_q, decoders_out = model([mu_q,logvar_q])
            else:
                print("UNIMODAL mode not implemented for this model")
                sys.exit()
        if gammas is None:
            gammas = gamma.detach().cpu()
            if not opt.classification:
                regression_pred = decoders_out[-1].detach().cpu()
            labels_all = labels.detach().cpu()
        else:
            gammas = torch.cat((gammas, gamma.detach().cpu()), dim=0)
            if not opt.classification:
                regression_pred = torch.cat((regression_pred, decoders_out[-1].detach().cpu()), dim=0)
            labels_all = torch.cat((labels_all, labels.detach().cpu()), dim=0)

    y_pred = gammas.argmax(dim = -1)
    if not opt.classification:
        preds, labels = denormalize(opt, regression_pred, labels_all)
        plot_regression(opt, preds, labels, plotdir, experimentID, mode)
        err =  F.mse_loss(torch.Tensor(preds), torch.Tensor(labels))
        return err
    else:
        unsup_preds = unsup_classification(labels_all, y_pred, plotdir, mode, experimentID, opt.num_clusters)
        plot_cm(opt,labels_all, unsup_preds, mode, experimentID, plotdir)
        return calc_acc(unsup_preds, labels_all)
