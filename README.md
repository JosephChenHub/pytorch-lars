# Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling
This is a non-official implementation of the optimizer Layer-wise Adaptive Rate Scaling (LARS) from ***Ginsburg, Boris, Igor Gitman, and Yang You. "Large batch training of convolutional networks with layer-wise adaptive rate scaling." ICLR'18***,  and ***LARGE BATCH OPTIMIZATION FOR DEEP LEARNING:TRAINING BERT IN 76 MINUTES. ICLR'20***

# Dependencies
- PyTorch >= 1.0


# Experiments of classification
- on mnist:

| Batch Size | epochs | warm-up | LR (LARS) |  acc. (LARS) | LR(SGD) | acc. (SGD) | LR(Adam) | acc. (Adam)|
| ------- | ------- | ------- | ------- | -------|-------|-------|-------|-------|
| 32 | 10 | 5 | 0.003| **99.52** |0.040 | 99.45 | 0.001 | 99.36 |
| 64 | 10 | 5 | 0.006| **99.54** |0.020 | 99.39 | 0.002 | 99.45 |
| 128 | 10 | 5 |0.006 | **99.54** |0.150 | 99.39 |0.005 | 99.26 |
| 512 | 10 | 5 | 0.012| **99.51** | 0.100 |99.29| 0.005 | 99.37 |
| 1024 | 10 | 5 |0.024 | **99.45** | 0.155 | 99.22 | 0.0155 | 99.30 |
| 2048 | 10 | 5 |0.030 | **99.42** | 0.050 | 98.71 | 0.0261 | 99.27 |
| 4096 | 10 | 5 |0.040 | **99.32** | 0.261 | 98.48| 0.0155 | 99.15 |

- on cifar-100:

| Backbone | Initialization | Augmentation | Optimizer | Hyper-parameters | Training time | Top1 Acc. |
| -------- | -------- | -------- | -------- |  -------- |-------- | -------- |
| [ResNet18](https://github.com/weiaicunzai/pytorch-cifar100) | PyTorch default| RandomCrop+HorizontalFlip| SGD | bz:128, lr:0.1, warmup:1,  multistep| - | 76.39| 
| ResNet18 | PyTorch default | RandomCrop+HorizontalFlip| LARS | bz:4096, max lr:0.04, warmup:20,one cycle policy| ~ 25 min. (4x 2080Ti) | 78.05 | 


- on ImageNet (tiny):

# Experiments of segmentation
- cityscapes:


 

