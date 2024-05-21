from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
from torch import FloatTensor, DoubleTensor
import pickle

from torchvision.transforms.functional import resize as tensor_resize
from torchvision import datasets
from torchvision.transforms import ToTensor

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return FloatTensor(y_cat)
class LatticeDataset(Dataset):
    def __init__(self,opt, mode="train"):
        self.opt = opt
        self.mode = mode
        self.datadir = opt.datadir
        self.Ndata = opt.num_1d_sample_points
        self.num_classes = opt.num_clusters
        self.rng_seed = opt.rng_seed
        self.img_dir = os.path.join(self.datadir,"data", "images_microscope")
        self.img_load_dir = os.path.join(self.datadir,"data", "images_load")
        self.curve_dir = os.path.join(self.datadir,"data", "force_disp_curves")
        self.param_dir = os.path.join(self.datadir,"data", "parameters")
        names = [f[:-4] for f in os.listdir(os.path.join(self.datadir,'data', 'parameters'))]
        labels = self.get_labels(names)
        #Stratify train/test split by build plate
        names_train, names_test, labels_train, labels_test  = train_test_split(names, labels, stratify=labels, test_size=0.1, random_state=self.rng_seed)
        names_train, names_val, labels_train, labels_val = train_test_split(names_train, labels_train, stratify=labels_train, test_size=0.1, random_state=self.rng_seed)
        
        if self.mode == "train":
            self.names = names_train
            self.labels = labels_train
        elif self.mode == "val":
            self.names = names_val
            self.labels = labels_val
        else: #test
            self.names = names_test
            self.labels = labels_test

        self.img_size = opt.img_size
        self.prep_lattice_data()
        
    def get_labels(self, names):
        labels = []
        for i in range(len(names)):
            if "-1" in names[i]:
                labels.append(0)
            else: # -2
                labels.append(1)
                
        return np.array(labels)

        
    def standardize_imgs(self, X):
        X = (X-X.min())/(X.max()-X.min())
        return X
    
    def prep_curves(self):
        c_min = 1e9
        c_max = -1e9
        min_strt_t = -1e9
        interpolated_curves = {}
        #Iterate over data to find largest minimum displacement collected to align all data by displacement
        for name in self.names:
            curve = np.load(os.path.join(self.curve_dir,"{}.npy".format(name)))
            disp = -curve[:,1]
            if disp[0] > min_strt_t:
                min_strt_t = disp[0]

        for name in self.names:
            curve = np.load(os.path.join(self.curve_dir,"{}.npy".format(name)))
            disp = -curve[:,1]
            force = -curve[:,2]
            xs = np.linspace(min_strt_t,1,self.Ndata)
            f = interp1d(disp, force)
            interpolated_curve = f(xs)
            interpolated_curves[name] = interpolated_curve
            if interpolated_curve.min() < c_min:
                c_min = interpolated_curve.min()
            if interpolated_curve.max() > c_max:
                c_max = interpolated_curve.max()

        #Normalize
        for k in interpolated_curves:
            curve = interpolated_curves[k]
            curve = curve - c_min
            curve = curve / (c_max-c_min)
            interpolated_curves[k] = curve
        return interpolated_curves

    def construct_tensors(self,interpolated_curves, x=460,y=460,x_crop=(150,610),y_crop=(285,745)):
        #param_tensor = np.zeros((len(names),len(param_keys)))
        img_tensor = np.zeros((len(self.names),x,y))
        curve_tensor = np.zeros((len(self.names), self.Ndata))

        for i in range(len(self.names)):
            name = self.names[i]
            img = np.load(os.path.join(self.img_dir,"{}.npy".format(name)))/255.0
            img = img.mean(axis=2)
            img_tensor[i] = img[x_crop[0]:x_crop[1],y_crop[0]:y_crop[1]]
            curve_tensor[i] = interpolated_curves[name]
        return img_tensor, curve_tensor   

    
    def augment(self, X, X2, y, names):
        #subsample
        X = np.concatenate([X[:,10:162,:152],X[:,162:314,152:304],X[:,10:162,:152],X[:,162:314,152:304]],axis=0)
        X2 = np.concatenate([X2,X2,X2,X2],axis=0)
        y = np.concatenate([y,y,y,y],axis=0)
        names = np.concatenate([names,names,names,names],axis=0)

        #rotate
        X = np.concatenate([X,np.flip(np.flip(X,axis=1),axis=2)],axis=0)
        X2 = np.concatenate([X2,X2],axis=0)
        y = np.concatenate([y,y],axis=0)
        names = np.concatenate([names,names],axis=0)

        return X, X2, y, names

    def resize_imgs(self, X):
        X_resize = np.zeros((len(X), self.img_size[0], self.img_size[1]))
        for i in range(len(X)):
            X_resize[i] = resize(X[i],(self.img_size[0], self.img_size[1]), preserve_range=True)
        return X_resize
        
    def prep_lattice_data(self):
        interpolated_curves = self.prep_curves()
        img_tensor, curve_tensor = self.construct_tensors(interpolated_curves)

        # standardize
        imgs = self.standardize_imgs(img_tensor)
        self.curves = curve_tensor

        # data augmentation
        if self.opt.augment_training:
            imgs, self.curves, self.labels, self.names = self.augment(imgs, self.curves, self.labels, self.names)
        
        #resize
        self.imgs = self.resize_imgs(imgs)
        self.labels = to_categorical(self.labels, self.num_classes)

  
    def __getitem__(self,idx):
        return {"ids":self.names[idx],  "inputs":[DoubleTensor(self.imgs[idx]).unsqueeze(0), DoubleTensor(self.curves[idx]).unsqueeze(0)], "class_lbls":FloatTensor(self.labels[idx])}

    def __len__(self):    
        return len(self.names)

class MNISTDataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode
        self.datadir = opt.datadir
        self.Ndata = opt.num_1d_sample_points
        self.num_classes = opt.num_clusters
        self.rng_seed = opt.rng_seed
        self.noiseLevel = opt.noiseLevel
        
        # Fetch/download data
        training_data = datasets.MNIST(
            root=self.datadir,
            train=True,
            download=True,
            transform=ToTensor()
        )
        
        test_data = datasets.MNIST(
            root=self.datadir,
            train=False,
            download=True,
            transform=ToTensor()
        )
        
        # Collect TEST labels, images, names
        labels_test = []
        images_test = []
        names_test  = []
        for i in range(0, len(test_data)):
            labels_test.append(test_data[i][1])
            images_test.append(test_data[i][0].double())
            names_test.append("test_%s" % i)

        # Collect TRAIN labels
        labels_training = []
        for i in range(0, len(training_data)):
            labels_training.append(training_data[i][1])
        
        # Create second modality (lines)
        xdata = np.linspace(0,1, self.Ndata)
        mod2_test = np.einsum('i,j->ij',labels_test,xdata) + np.random.normal(0,self.noiseLevel, (len(labels_test), self.Ndata))
        mod2_training = np.einsum('i,j->ij',labels_training,xdata) + np.random.normal(0,self.noiseLevel, (len(labels_training), self.Ndata))

        #Stratify train/validate split by labels
        indices_training = range(0,len(training_data)) #serves as initial names_train
        indices_train, indices_val, labels_train, labels_val = train_test_split(indices_training, labels_training, stratify=labels_training, test_size=0.1, random_state = self.rng_seed)
        
        # Collect images and names for training
        images_train = []
        names_train = []
        mod2_train = []
        for i in indices_train:
            images_train.append(training_data[i][0].double())
            names_train.append("train_%s" % i)
            mod2_train.append(mod2_training[i])
        
        # Collect images and names for validation
        images_val = []
        names_val = []
        mod2_val = []
        for i in indices_val:
            images_val.append(training_data[i][0].double())
            names_val.append("train_%s" % i)
            mod2_val.append(mod2_training[i])
            
        if self.mode == "train":
            self.names = names_train
            self.labels = to_categorical(np.array(labels_train), self.num_classes)
            self.images = images_train
            self.mod2 = mod2_train
        elif self.mode == "val":
            self.names = names_val
            self.labels = to_categorical(np.array(labels_val), self.num_classes)
            self.images = images_val
            self.mod2 = mod2_val
        else: #test
            self.names = names_test
            self.labels = to_categorical(np.array(labels_test), self.num_classes)
            self.images = images_test
            self.mod2 = mod2_test
        
        self.img_size = opt.img_size

    def __getitem__(self,idx):
        return {"ids":self.names[idx],  "inputs":[self.images[idx], torch.DoubleTensor(self.mod2[idx]).unsqueeze(0)], "class_lbls": self.labels[idx]}
        
    def __len__(self):
        return len(self.names)
