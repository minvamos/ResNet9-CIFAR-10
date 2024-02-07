# ResNet-9 Model Training and Evaluation

This repository provides tools for training and evaluating ResNet-9 models. It includes scripts to train ResNet-9 model  on a CIFAR-10 dataset and then evaluate/test the model.

No pre-learning model was used.

## Table of Contents
1. [Model](#model)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Result](#result)

## Model
![Layers](https://github.com/minvamos/ResNet9-CIFAR-10/assets/122091776/3ed83ff9-a747-4002-913e-b930e111e04a)

The above model is implemented on [**Resnet.py**](https://github.com/minvamos/ResNet9-CIFAR-10/blob/main/src/resnet.py). It has fewer layers and is lighter than the regular Resnet model.

## Setup 

**Note:** This repository ran in an Apple Silicon MacOS environment. So it is 'mps' based, it does not run in a GPU environment.

1. Clone this repository.

```bash
git clone https://github.com/minvamos/ResNet9-CIFAR-10.git

cd ResNet9-CIFAR-10
```

2. Install required packages.

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data in the appropriate directory.
2. Train and test the model.

```bash
python main_ResNet9.py
```

## Result 
![graph](https://github.com/minvamos/ResNet9-CIFAR-10/assets/122091776/aeae9483-28fc-4aa8-a758-16aea7011ace)
Running on three random seeds, with an accuracy average of 90.13%
