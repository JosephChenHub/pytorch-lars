# Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling
This is a non-official implementation of the optimizer Layer-wise Adaptive Rate Scaling (LARS) from ***Ginsburg, Boris, Igor Gitman, and Yang You. "Large batch training of convolutional networks with layer-wise adaptive rate scaling." ICLR'18*** 

# Dependencies
- PyTorch >= 1.0


# Experiments of classification
- on mnist:

| Batch Size | epochs | warm-up | LR (LARS) |  acc. (LARS) | LR(SGD) | acc. (SGD) | LR(Adam) | acc. (Adam)|
| ------- | ------- | ------- | ------- | -------|-------|-------|-------|-------|
| 32 | 10 | 5 | 4.5| **99.49** |0.040 | 99.45 | 0.001 | 99.36 |
| 64 | 10 | 5 | 7| **99.48** |0.020 | 99.39 | 0.002 | 99.45 |
| 128 | 10 | 5 |12 |**99.50** |0.150 | 99.39 |0.005 | 99.26 |
| 512 | 10 | 5 | 20|**99.48** | 0.100 |99.29| 0.005 | 99.37 |
| 1024 | 10 | 5 |14 | **99.41** | 0.155 | 99.22 | 0.0155 | 99.30 |
| 2048 | 10 | 5 |20 | **99.39** | 0.050 | 98.71 | 0.0261 | 99.27 |
| 4096 | 10 | 5 |20 | **99.07** | 0.261 | 98.48| 0.0155 | 99.15 |

- on cifar-100:


- on ImageNet (tiny):

# Experiments of segmentation
- cityscapes:


 

