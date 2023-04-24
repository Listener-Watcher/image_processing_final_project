# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
[![DOI](https://zenodo.org/badge/241184407.svg)](https://zenodo.org/badge/latestdoi/241184407)


### Blog post with full documentation: [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)

### See also [PyTorch Implementation for BYOL - Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://github.com/sthalles/PyTorch-BYOL).

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python
## add new args: noise, prob
noise:jitter,flip,label,gaussian,gray,none
prob:between 0 and 1
gaussian std mean changes need to modify gaussian blur file in the SimCLR/data_aug/gaussian_blur.py
## dataset now become Medmnist dataset, dataset-name used default one.
sepcify --download for downloading dataset
$ python run.py -data ./datasets --dataset-name 

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

For 16-bit precision GPU training, there **NO** need to to install [NVIDIA apex](https://github.com/NVIDIA/apex). Just use the ```--fp16_precision``` flag and this implementation will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).

## Feature Evaluation
run python evaluate_cl.py or python evaluate_cl3.py for evaluation.
