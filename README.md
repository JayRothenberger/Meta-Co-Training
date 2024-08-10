# Meta Co Training

This branch is unfinished.  Things may not work correctly or reflect the advertized functionality of performance.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meta-co-training-two-views-are-better-than/semi-supervised-image-classification-on-2)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-2?p=meta-co-training-two-views-are-better-than)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meta-co-training-two-views-are-better-than/semi-supervised-image-classification-on-1)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-1?p=meta-co-training-two-views-are-better-than)

ArXiv paper: [https://arxiv.org/abs/2311.18083](https://arxiv.org/abs/2311.18083)

In many practical computer vision scenarios unlabeled data is plentiful, but labels are scarce and difficult to obtain. As a result, semi-supervised learning which leverages unlabeled data to boost the performance of supervised classifiers have received significant attention in recent literature. One major class of semi-supervised algorithms is co-training. In co-training two different models leverage different independent and sufficient "views" of the data to jointly make better predictions. During co-training each model creates pseudo labels on unlabeled points which are used to improve the other model. We show that in the common case when independent views are not available we can construct such views inexpensively using pre-trained models. Co-training on the constructed views yields a performance improvement over any of the individual views we construct and performance comparable with recent approaches in semi-supervised learning, but has some undesirable properties. To alleviate the issues present with co-training we present Meta Co-Training which is an extension of the successful Meta Pseudo Labels approach to two views. Our method achieves new state-of-the-art performance on ImageNet-10% with very few training resources, as well as outperforming prior semi-supervised work on several other fine-grained image classification datasets. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Experiments](#reproducing-experiments)
- [Experimental Results](#experimental-results)
- [Features](#features)
- [Roadmap](#roadmap)
- [License](#license)

## Installation

To install the package first clone this repository

- `git clone https://github.com/JayRothenberger/Meta-Co-Training`

Then install the package in the folder using pip

- `pip install -e Meta-Co-Training/meta-cotraining`

Finally, install the unlisted clip package from OpenAI

- `pip install pip install git+https://github.com/openai/CLIP.git`

## Usage

Each view requires its own GPU to train on.  To facilitate this we will need the NCCL backend for torch.distributed, and we will also assume your processes are launched with torchrun.

```python
# standard library imports
import argparse
from copy import deepcopy as copy
import os
# installed imports
import torch
import torchvision
import wandb
# local code imports
from mct.MCT import MetaCoTrainingModel
from mct.image_models import IMAGE_DISTANCES, IMAGE_TRANSFORMS
from mct.models import MLPProbe
from mct.utils import subset_npercent_dataset

def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()
```

Meta Co-Training requires multiple independent and sufficient views of the classification problem to be effective.  We store each view as its own dataset in the general case, but in the example below views are constructed from representations so the dataset objects are identical.

```python
# connect to the process group
setup()
# specify the device for the current process
device = int(os.environ['RANK']) % torch.cuda.device_count()

torch.manual_seed(13)
torch.cuda.set_device(device)

views = ['DINOv2', 'CLIP']
# the view that will be trained on this device
view = views[int(os.environ['RANK']) % len(views)]

# the index in the views list will correspond to the index in these lists
# these lists will store the dataset objects for each view
trains = []  # training views
unlbls = []  # unlabeled views
vals = []  # validation views
```

In this example we will avoid preprocessing the dataset embeddings for the different models by instead performing an inference step with the model one each image instance.  This is more expensive, but less complicated than building a dataset of the embeddings.  It also allows us to fine-tune the representation model if we wish.

```python
# for each view we create a dataset object and specify the correct transform
for view in views:
    dataset = torchvision.datasets.ImageNet(args.dataset_path, split='train', transform=IMAGE_TRANSFORMS[view])
    val = torchvision.datasets.ImageNet(args.dataset_path, split='val', transform=IMAGE_TRANSFORMS[view])

    # split the data
    train, unlbl = subset_npercent_dataset(dataset, percent=args.train_size * 100)

    # append the dataset objects to the view lists
    trains.append(train)
    unlbls.append(unlbl)
    vals.append(val)

num_classes = 1000
```

The package has already specified a dictionary that connects models' names to their expected preprocessing, and pretrained weights.

```python
# for each view, get the correct foundation model
models = [IMAGE_DISTANCES[view]().model for view in views]
# send each model to the current device
for model in models:
    model.to(device)
```

We append an MLP head after the frozen representation generated by the model which will do our learning and meta learning.  Below warmup and regular training are separated, but this is not necessary and just illustrates that they are logically independent in the algoritm design.

```python
# append the MLP head on top of the representation model
models = [MLPProbe(model, shape, num_classes) for shape, model in zip(shapes, models)]
# construct the MCT model object which implements a custom training loop
MCTModel = MetaCoTrainingModel(models)

# supervised warmup
states = MCTModel.train(warmup_epochs, 
                    warmup_epochs + 1, 
                    trains, 
                    unlbls, 
                    vals, 
                    vals, 
                    checkpoint_path=f'./chkpts/{view}_chkpt', 
                    batch_size=batch_size, 
                    log_interval=100
                    )
# preparation stage in which the model does not alter embedder weights
states = MCTModel.train(epochs, 
                    0, 
                    trains, 
                    copy(unlbls), 
                    copy(vals), 
                    copy(vals), 
                    checkpoint_path=f'./chkpts/{view}_chkpt', 
                    batch_size=batch_size, 
                    log_interval=100
                    )
```

## Reproducing Experiments

This branch of the repository uses a more user-friendly and polished implementation of the meta co-training algorithm that has been extended to support more than two views, faster training and validation, calibrated teacher model predictions, gradient accumulation, and a greater number of image foundation models.  As such, the randomness, floating point representation, foundation model sizes, and some minor algorithmic choices like MLP size and logit temperature are not the same as the original implementation.  The original implementation can still be found in the legacy branch.  That being said, we find the results that this code yields are nearly identical to the legacy branch with the following script:

1. `mct_e2e.sh` which runs `MCT_e2e.py`


## Experimental Results

Results for experiments beyond those presented in our paper are coming soon.

## Roadmap
✅ MCTModel Object \
&nbsp;&nbsp;&nbsp;&nbsp;✅ Support More than Two Views\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ (Auto) Differentiable AllGather operation\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ One model per GPU\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ Multiple GPUs per model\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Automatic Mixed Precision Support\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Gradient Accumulation\
✅ Calibrated Teacher Model Predictions\
✅ Full-Parameter Fine-Tuning\
⬜️ Support More Foundation Models for View Construction\
&nbsp;&nbsp;&nbsp;&nbsp;✅ CLIP\
&nbsp;&nbsp;&nbsp;&nbsp;✅ DINOv2\
&nbsp;&nbsp;&nbsp;&nbsp;✅ SwAV\
&nbsp;&nbsp;&nbsp;&nbsp;✅ EsViT\
&nbsp;&nbsp;&nbsp;&nbsp;✅ MAE\
&nbsp;&nbsp;&nbsp;&nbsp;✅ BLIP\
&nbsp;&nbsp;&nbsp;&nbsp;✅ SigLIP\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Heira\
&nbsp;&nbsp;&nbsp;&nbsp;✅ ALIGN\
&nbsp;&nbsp;&nbsp;&nbsp;✅ OWLv2\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ EVA-01\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ EVA-02\
⬜️ Experimental Evaluation With Multiple Views\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ ...


## License
Copyright 2024 Jay Rothenberger (jay.c.rothenberger@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.