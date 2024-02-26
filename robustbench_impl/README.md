# RobustBench Submission

This directory is the RobustBench submission of MixedNUTS and is meant to be used independently from our main code.
This implementation is compatible with the RobustBench API.

Please see `script.sh` for scripts and instructions for running the code.
In addition to `robustbench` and its dependencies, this implementation only requires installing `click` via pip.


# Paper Information

- **Paper Title**: MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers
- **Paper URL**: https://arxiv.org/abs/2402.02263
- **Paper Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi


# Leaderboard Claim(s)

## Model 1

- **Architecture**: ResNet-152 + RaWideResNet-70-16
- **Dataset**: CIFAR-10
- **Threat model**: Linf
- **eps**: 8 / 255
- **Clean accuracy**: 95.19 %
- **AutoAttack accuracy**: 70.08 %
- **Best-known robust accuracy**: 69.71 %
- **Additional data**: True.
  The accurate base classifier was pre-trained on ImageNet; the robust base classifier used 50M synthetic images
- **Evaluation method**: AutoAttack; Adaptive AutoAttack
- **Checkpoint and code**: The code is available [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl).
  Please see [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl#model-checkpoints)
  for the download links and instructions for the model checkpoints

## Model 2

- **Architecture**: ResNet-152 + WideResNet-70-16
- **Dataset**: CIFAR-100
- **Threat model**: Linf
- **eps**: 8 / 255
- **Clean accuracy**: 83.08 %
- **AutoAttack accuracy**: 41.91 %
- **Best-known robust accuracy**: 41.80 %
- **Additional data**: True.
  The accurate base classifier was pre-trained on ImageNet; the robust base classifier used 50M synthetic images
- **Evaluation method**: AutoAttack; Adaptive AutoAttack
- **Checkpoint and code**: The code is available [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl).
  Please see [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl#model-checkpoints)
  for the download links and instructions for the model checkpoints

## Model 3

- **Architecture**: ConvNeXt V2-L + Swin-L
- **Dataset**: ImageNet
- **Threat model**: Linf
- **eps**: 4 / 255
- **Clean accuracy**: 81.48 %
- **AutoAttack accuracy**: 58.62 %
- **Best-known robust accuracy**: 58.50 %
- **Additional data**: True.
  The accurate base classifier was pre-trained on ImageNet-21k
- **Evaluation method**: AutoAttack; Adaptive AutoAttack
- **Checkpoint and code**: The code is available [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl).
  Please see [here](https://github.com/Bai-YT/MixedNUTS/tree/main/robustbench_impl#model-checkpoints)
  for the download links and instructions for the model checkpoints


# Model Checkpoints

MixedNUTS is a training-free method that has no additional neural network components other than its base classifiers.

All robust base classifiers used in the main results of our paper are available on [RobustBench](https://robustbench.github.io)
and can be downloaded automatically via the RobustBench API.

Here, we provide the download links to the standard base classifiers used in the main results.

| Dataset   | Link  |
|-----------|-------|
| CIFAR-10  | [Download](http://172.233.227.28/base_models/cifar10/cifar10_std_rn152.pt)    |
| CIFAR-100 | [Download](http://172.233.227.28/base_models/cifar100/cifar100_std_rn152.pt)  |
| ImageNet  | [Download](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt)  |

After downloading the accurate base classifiers, create a `base_models` directory and organize as follows:
```
base_models
│
└───cifar10
│   └───cifar10_std_rn152.pt
│   
└───cifar100
    └───cifar100_std_rn152.pt
│   
└───imagenet
    └───imagenet_std_convnext_v2-l_224.pt
```


# Model Zoo:

- [x] I want to add my models to the Model Zoo (check if true).
- [x] <del>I use an architecture that is included among those
  [here](https://github.com/RobustBench/robustbench/tree/master/robustbench/model_zoo/architectures)
  or in `timm`. If not,</del> I added the link to the architecture implementation so that it can be added.
- [x] I agree to release my model(s) under MIT license (check if true).
