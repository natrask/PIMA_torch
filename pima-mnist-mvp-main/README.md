## PIMA: Physics-Informed Multimodal Autoencoder
#### SAND2022-16315 O

For algorithm details, refer to our paper preprint: https://arxiv.org/abs/2202.03242.

For implementation details, contact cmarti5@sandia.gov.


## Running PIMA

To train a multimodal PIMA model (in a python environment setup for PIMA):

```bash
python main.py configs/mnist_multimodal_example.yaml
```

To train a unimodal model given a trained multimodal model (whose path is set in the config file):

```bash
python main.py configs/mnist_unimodal_example.yaml
```

## Python Environment Setup

For reproducibility, we use PyTorch settings that rely on the following environment variable setting.  This can be added to your .bashrc file:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

To setup an environment for PIMA:

Tested with Python 3.7.6 using pip to install requirements. Modify torch and torchvision versions in requirements.txt to reflect the correct CUDA setup for your machine.  Code was tested with CUDA 11.1 version of PyTorch.

```bash
pip install -r requirements.txt
```

Notes: The config file includes optional Weights & Biases (wandb.ai) settings that require a wandb account.  The default behavior does NOT use wandb.

The code will automatically download the pytorch version of the MNIST dataset, and will write output to the directory defined in the config file as "savedir".
