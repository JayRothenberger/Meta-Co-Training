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

To install this repository first clone this repository

- `git clone https://github.com/JayRothenberger/Meta-Co-Training`

Then install the package in the folder using pip

- `pip install -e Meta-Co-Training/meta-cotraining`

## Usage

TODO

## Reproducing Experiments

Using the examples present in the examples directory, models can be trained by executing a sequence of scripts.  

1. `mcp_prep.sh`
2. `mcp_pretrain.sh`
3. `mcp_fpft.sh`

## Experimental Results

TODO

## Roadmap

⬜️ Finish the ReadMe\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Start writing it\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Finish writing it

## License
Copyright 2024 Jay Rothenberger (jay.c.rothenberger@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.