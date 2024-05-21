import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import wandb
import os 
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch
from .utils import get_device
from .VAE import reparameterize
import sys

def plot_cm(opt, y_test, perm_pred, mode, run_name, plotdir):
    cm = confusion_matrix(y_test, perm_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    plt_fname = os.path.join(plotdir, "CM_{}_{}.png".format(mode, run_name))
    plt.savefig(plt_fname)
    if opt.use_wandb:
        wandb.log({"Confusion_matrix_{}".format(mode):wandb.Image(plt_fname)})
    plt.close()


def plot_regression(opt, preds, labels, plotdir, experimentID, mode):
    plt.figure()
    plt.scatter(labels, preds, s=3)
    xmin = min(labels.min(), preds.min())
    xmax = max(labels.max(), preds.max())
    plt.plot([xmin,xmax],[xmin,xmax], color="black")
    plt.xlabel("ground truth")
    plt.ylabel("predicted values")
    outname = os.path.join(plotdir, "regression_result{}_{}.png".format(experimentID, mode))
    plt.savefig(outname)
    if opt.use_wandb:
        wandb.log({"Regression_result_{}".format(mode):wandb.Image(outname)})
    plt.close()


def plot_modalities(opt, batch, decoders_out, plotdir, experimentID, itr, mode, modality_idxs_to_plot=[]):
    names = batch["ids"] 
    XS = batch["inputs"] 
    if modality_idxs_to_plot==[]:
        modality_idxs_to_plot = range(len(XS))
        
    for i in modality_idxs_to_plot:
        if isinstance(decoders_out[i], tuple): #Expert model for lattices has tuple output
            decoders_out[i]=decoders_out[i][0]
        batch_sz = XS[i].shape[0]

        #TODO: generalize for multichannel imgs
        if len(XS[i].shape) == 4: #2D data
            for j in range(1):#len(names)):
                fig = plt.figure()
                ax = fig.add_subplot(121)
                plt.title("Input X{}".format(i))
                plt.grid(False)
                plt.imshow(XS[i][j].squeeze().detach().cpu().numpy(), cmap='Greys')
                ax = fig.add_subplot(122)
                plt.title("X{} decoded".format(i))
                plt.grid(False)
                plt.imshow(decoders_out[i][j].squeeze().detach().cpu().numpy(), cmap='Greys')
                outname = os.path.join(plotdir, "X{}_{}_{}_{}.png".format(i, names[j], experimentID, itr))
                plt.savefig(outname)
                if opt.use_wandb:
                    wandb.log({"X{}_{}".format(i, mode):wandb.Image(outname)})
                plt.close()
                
        elif len(XS[i].shape) == 3: #1D data
            _, ch, _ = XS[i].shape
            for j in range(1):
                fig = plt.figure()
                ax = fig.add_subplot(121)
                plt.title("Input X{}".format(i))
                for k in range(ch):
                    plt.plot(range(len(XS[i][j][k])), XS[i][j][k].detach().cpu().numpy(), c= "black")
                ax = fig.add_subplot(122)
                plt.title("X{} decoded".format(i))

                for k in range(ch):
                    plt.plot(range(len(decoders_out[i][j][k])), decoders_out[i][j][k].detach().cpu().numpy(), c="green")
                    outname = os.path.join(plotdir, "X{}_{}_{}_{}.png".format(i, names[j], experimentID, itr))
                plt.savefig(outname)
                if opt.use_wandb:
                    wandb.log({"X{}_{}".format(i, mode):wandb.Image(outname)})
                plt.close()
        else: #misc data
            for j in range(1):
                fig = plt.figure()
                ax = fig.add_subplot(121)
                plt.title("Input X{}".format(i))

                if XS[i][j].shape[0] == 1:
                    plt.scatter([0], [XS[i][j].detach().cpu()], c= "black")
                else:
                    plt.scatter(range(len(XS[i].squeeze()[j])), XS[i].squeeze()[j].detach().cpu().numpy(), c= "black")
                ax = fig.add_subplot(122)
                plt.title("X{} decoded".format(i))

                if opt.n_params == 1:
                    plt.scatter(range(opt.n_params), [decoders_out[i][j].detach().cpu()], c="green")
                else:
                    plt.scatter(range(opt.n_params), decoders_out[i][j].detach().cpu().numpy(), c="green")
                outname = os.path.join(plotdir, "X{}_{}_{}.png".format(i, names[j], experimentID, itr))
                plt.savefig(outname)
                if opt.use_wandb:
                    wandb.log({"X{}_{}".format(i, mode):wandb.Image(outname)})
                plt.close()


def viz_batch(opt, dataloader, model, itr, unimodal_model=None, mode="train"):
    with torch.no_grad():
        model.eval()
        device = get_device(opt.gpu)
        experimentID = opt.experimentID
        plotdir = os.path.join(opt.savedir, str(opt.name), str(opt.experimentID), 'plots')
        os.makedirs(plotdir, exist_ok=True)

        for i, batch in enumerate(dataloader):
            XS = [x.to(device) for x in batch["inputs"]]
            names = batch["ids"]

            if unimodal_model is None:
                gamma, pi, mu_q, logvar_q, decoders_out = model(XS)
            else:
                mu_q, logvar_q = unimodal_model(XS)
                model.unimodal = True
                if opt.dataset == "lattice":
                    gamma, pi, mu_q, logvar_q, decoders_out = model([mu_q,logvar_q])
                elif opt.dataset == "mnist":
                    gamma, pi, mu_q, logvar_q, decoders_out = model([mu_q,logvar_q])
                else:
                    print("UNIMODAL mode not implemented for this model")
                    sys.exit()

            if opt.classification:
                #Plot clusters with ground truth labels as color
                labels = batch["class_lbls"]
                labels = labels.argmax(axis=-1)            
            else:
                #Plot clusters with predicted class as color
                labels = gamma.argmax(axis=-1)            
                #labels = torch.zeros(len(XS[0]), dtype=int)

            zs = reparameterize(mu_q, logvar_q)

            outname = os.path.join(plotdir, "X2_{}_{}.png".format(experimentID, itr))
            if opt.dataset =="lattice" and opt.data_driven==False:
                plt.figure()
                colors = ["blue","orange"]
                for j in range(len(names)):
                    plt.plot(XS[1].squeeze().detach().cpu().numpy()[j], c=colors[labels[j]])

                for j in range(opt.num_clusters):
                    plt.plot(decoders_out[1][0][j].detach().cpu().numpy(), c='red')
                plt.savefig(outname)
                if opt.use_wandb:
                    wandb.log({"X2_{}".format(mode):wandb.Image(outname)})
                plt.close()

                plot_modalities(opt, batch, decoders_out, plotdir, experimentID, itr, mode, modality_idxs_to_plot=[0])
            
            elif opt.dataset == "mnist" and opt.data_driven == False:
                plt.figure()
                colors = ["red", "orange", "yellow","green","blue","purple", "pink", "aqua", "magenta","gray"]
                
                for j in range(opt.num_clusters):
                    plt.plot(decoders_out[1][0][j].detach().cpu().numpy(), c=colors[j])
                plt.savefig(outname)
                if opt.use_wandb:
                    wandb.log({"X2_{}".format(mode):wandb.Image(outname)})
                plt.close()
                
                plot_modalities(opt, batch, decoders_out, plotdir, experimentID, itr, mode, modality_idxs_to_plot=[0])
                
            else:
                plot_modalities(opt, batch, decoders_out, plotdir, experimentID, itr, mode)
            

            plot_clusters(opt, zs.detach().cpu(), logvar_q.detach().cpu(), labels, model.mu_gmm.detach().cpu(), model.logvar_gmm.detach().cpu(), opt.num_clusters, plotdir, experimentID, itr, mode=mode)
            #plot one batch
            break

def plot_clusters(opt, zs, logvar_q, labels, centers, logvar, n_clusters, plotdir, run_name, it, mode="train"):
    dim = zs.shape[1]
    zs = zs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    fig = plt.figure(constrained_layout=True,figsize=[15,9])
    colors = np.array(["blue", "orange", "yellow", "green", "red", "purple", "pink","aqua","magenta","gray"])
    
    if dim >= 3: #TODO multiple plots for 4d encoding dim
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for cls in range(n_clusters):
        if dim == 1:
            ax.scatter(centers[:, 0], np.zeros_like(centers[:,0]), c='black', s=150)
            ax.scatter(zs[:, 0],np.zeros_like(zs[:, 0]), c=colors[labels])
        if dim == 2:
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=150)
            ax.scatter(zs[:, 0], zs[:, 1], c=colors[labels])
        if dim >= 3:
            ax.scatter(centers[:,0],centers[:,1], centers[:,2],c='black',s=150)
            ax.scatter(zs[:, 0], zs[:, 1], zs[:, 2], c=colors[labels])

    plt_fname = os.path.join(plotdir,"clusters_{}_{}_{}.png".format(mode,it, run_name))
    plt.savefig(plt_fname)
    if opt.use_wandb:
        wandb.log({"clusters_{}".format(mode):wandb.Image(plt_fname)})
    plt.close()
