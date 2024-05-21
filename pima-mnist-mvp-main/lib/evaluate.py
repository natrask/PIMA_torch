import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

def rl2_err(preds, labels):
    return torch.sqrt(((preds-labels)**2).sum())/torch.sqrt((labels**2).sum())

def unsup_classification(y_test, predclass, plotdir, mode, run_name, num_clusters=1):
    y_test = y_test.numpy()
    predclass = predclass.numpy()
    err = np.zeros((num_clusters,num_clusters))

    for j in range(num_clusters):
        for i in range(num_clusters):
            #Permute cluster labels
            lbls = np.ones((predclass[predclass==j]).shape)*i

            if len(lbls) > 0:
                err[j][i] =  (1.0*(lbls!=y_test[predclass==j])).sum()
            else:
                err[j][i] = 0.0

    row_ind, col_ind = linear_sum_assignment(err)
    perm_pred = predclass.copy()
    for i in range(num_clusters):
        perm_pred[predclass==i] = col_ind[i]
    return perm_pred

def calc_acc(preds, lbls):
    acc = (1.0*(torch.Tensor(preds)==lbls)).sum()/len(preds)
    return acc
