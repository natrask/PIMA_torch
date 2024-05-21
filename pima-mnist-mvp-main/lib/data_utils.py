from pytorch_datasets.bfp_datasets import *
import sys

def denormalize(opt,regression_pred, labels_all):
    preds = regression_pred.cpu().numpy()
    labels = labels_all.cpu().numpy()
    if opt.normalization_parameters:
        c0 = opt.normalization_parameters[-2]
        c1 = opt.normalization_parameters[-1]
        labels = (labels - c1)/c0
        preds = (preds - c1)/c0
    return preds, labels

#Initialize datasets
def init_dataloaders(opt):
    print("Loading data")
    if opt.dataset == "lattice":
        ds_type = LatticeDataset
    elif opt.dataset == "mnist":
        ds_type = MNISTDataset
    else:
        print("DATASET {} NOT IMPLEMENTED".format(opt.dataset))
        sys.exit()

    if opt.use_wandb and opt.use_artifact: #Download data/path from w&b
        import wandb 
        artifact = wandb.use_artifact(opt.use_artifact)
        opt.datadir = artifact.download('./artifacts/{}'.format(opt.dataset))

    train_dataset = ds_type(opt)
    val_dataset = ds_type(opt, mode="val")
    test_dataset = ds_type(opt, mode="test")

    batch_size = opt.batch_size if opt.batch_size <= len(train_dataset) else len(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
    batch_size = opt.batch_size if opt.batch_size <= len(val_dataset) else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
    batch_size = opt.batch_size if opt.batch_size <= len(test_dataset) else len(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
