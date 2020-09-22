from networks import *
from torchvision import datasets, transforms
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
#from resnet_official import resnet18, resnet34, resnet50, resnet101, resnet152

import torch.nn as nn
from cutout import Cutout
#from AutoAugment.autoaugment import CIFAR10Policy
#from fast_autoaugment.FastAutoAugment.augmentations import *
#from fast_autoaugment.FastAutoAugment.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

def get_network(name, num_classes=-1, bn_layer=nn.BatchNorm2d):
    if name == 'LeNet':
        return LeNet()
    elif name == 'resnet18':
        assert num_classes > 0
        return resnet18(num_classes=num_classes, norm_layer=bn_layer)
    elif name == 'resnet34':
        return resnet34(num_classes=num_classes, norm_layer=bn_layer)
    elif name == 'resnet50':
        return resnet50(num_classes=num_classes, norm_layer=bn_layer)
    elif name == 'resnet101':
        return resnet101(num_classes=num_classes, norm_layer=bn_layer)
    else:
        raise Exception("Unkown network:", name)



def get_dataset(name):
    if name == 'MNIST':
        return datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), \
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), 10
    elif name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

        return trainset, testset, 10
    elif name == 'CIFAR100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #CIFAR10Policy(),
            #Augmentation(autoaug_paper_cifar10()),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #Cutout(1, 8)
        ])
        cifar100_train = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cifar100_test = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

        return cifar100_train, cifar100_test, 100

    elif name == 'ImageNet':
        pass
    else:
        raise Exception("Unkown dataset:", name)

