"""MNIST example.

Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import random
import os
import warnings
from torch.utils.data.distributed import DistributedSampler

from optimizers import Lamb, log_lamb_rs
from optimizers import LARS
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time

from utils import get_network, get_dataset
from lr_scheduler import WarmUpLR
from lr_finder import LRFinder

import logging


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--net', type=str, default='LeNet',
                    help='network (default: LeNet)')
parser.add_argument('--eval', type=bool, default=False, help='eval the network')
parser.add_argument('--checkpt', type=str, help="checkpoint")
parser.add_argument('--grid_n', type=int, default='10',
                    help='lr grid search nums. (default: 10)')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'],
                    help='dataset (default: MNIST)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'lars', 'adam', 'sgd'],
                    help='which optimizer to use')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr_finder', type=int, default=0)
parser.add_argument('--mixup', type=int, default=0)
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--warmup', type=int, default=5, help='warmup epochs')
parser.add_argument('--lr', type=str, default="0.1,0.1", metavar='LR',
                    help='learning rate (default: 0.0025)')
parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                    help='weight decay (default: 0.01)')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    print("World size:", args.world_size)

    ngpus_per_node = torch.cuda.device_count()
    args.distributed = ngpus_per_node > 1


    main_worker(args.local_rank, ngpus_per_node, args)






def train_one_epoch(logger, args, model, device, train_loader, optimizer, epoch, event_writer, criterion, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    #progress = ProgressMeter(
    #    len(train_loader),
    #    [batch_time, data_time, losses, top1, top5],
    #    prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        if args.mixup:
            indices = torch.randperm(data.size(0))
            shuffled_data = data[indices]
            shuffled_targets = target[indices]

            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            lam = max(lam, 1. - lam)
            assert 0.0 <= lam <= 1.0, lam
            data = data * lam + shuffled_data * (1 - lam)

            output = model(data)
            loss = lam * criterion(output, target) + (1 - lam)* criterion(output, shuffled_targets)
        else:
            output = model(data)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # we only compute the metrics of a  batch on each gpu
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr scheduler
        if epoch <= int(args.epochs * 1.0) :
            scheduler.step()
        else:
            optimizer.param_groups[0]['lr'] = 8e-5


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0 and (batch_idx % args.log_interval == 0):
            step = batch_idx * len(data) + (epoch-1) * len(train_loader.dataset)
            log_lamb_rs(optimizer, event_writer, step)
            event_writer.add_scalar('loss', loss.item(), step)
            lr = optimizer.param_groups[0]['lr']
            event_writer.add_scalar('lr', lr, step)

            ratio_0 = optimizer.param_groups[0]['ratio_0']
            ratio_1 = optimizer.param_groups[0]['ratio_1']
            event_writer.add_scalars('ratios', {"first":ratio_0, "last":ratio_1}, step)
            #progress.display(batch_idx)
            logger.info("Training epoch:%s, lr:%.6f, loss:%.6f, acc1:%.4f, acc5:%.4f"%(epoch, lr, losses.avg, top1.avg, top5.avg))

    return losses.avg, top1.avg, top5.avg

@torch.no_grad()
def test_one_epoch(logger, args, model, device, test_loader, event_writer:SummaryWriter, epoch, criterion):
    model.eval()
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))


        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))


    # reduce the metrics
    losses_avg = reduce_tensor(torch.tensor([losses.avg]).cuda(args.gpu)).item()
    top1_avg = reduce_tensor(torch.tensor([top1.avg]).cuda(args.gpu)).item()
    top5_avg = reduce_tensor(torch.tensor([top5.avg]).cuda(args.gpu)).item()

    if dist.get_rank() == 0:
        logger.info(' Testing epoch: {}, Average loss: {:.4f}, Top1-Acc: {:.4f}, Top5-Acc: {:.4f}'.format(
            epoch, losses_avg, top1_avg, top5_avg))

    return losses_avg, top1_avg, top5_avg




def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        print("use distributed environment!")
        #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                        world_size=args.world_size, rank=args.local_rank)
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    device = torch.device("cuda:%s"%args.gpu)

    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_data, test_data, num_classes = get_dataset(args.dataset)
    args.num_classes = num_classes

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_data)
        ngpus = torch.cuda.device_count()
        train_batch_size = args.batch_size // ngpus
        test_batch_size = args.test_batch_size // ngpus
        val_sampler = DistributedSampler(test_data, shuffle=False) # distributed sampler
    else:
        train_batch_size = args.batch_size
        test_batch_size = args.test_batch_size


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=val_sampler,
        **kwargs)

    logger = None
    if dist.get_rank() == 0:
        logger = logging.getLogger()
        logger.setLevel('DEBUG')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler('%s_%s_%s.log'%(args.optimizer, args.dataset, args.batch_size))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

    writer = None
    if dist.get_rank() == 0:
        writer = SummaryWriter(comment="_optim_%s_%s_%s"%(args.optimizer, args.dataset, args.batch_size))

    lo, hi = args.lr.split(",")
    grid = np.linspace(float(lo), float(hi), args.grid_n)
    if dist.get_rank() == 0:
        logger.info("world_size:%s"%dist.get_world_size())
        logger.info("fix the seed:%s" % args.seed)
        logger.info("Use optimizer:%s, dataset:%s, batch size:%s"% (args.optimizer, args.dataset, args.batch_size))
        logger.info("search lr from: %s to %s with number:%s" %(lo, hi, args.grid_n))
        logger.info("Use mixup ? %s,  beta:%s " % (args.mixup, args.mixup_alpha))


    op_loss, op_acc = 0, 0
    op_lr = 0
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    for lr in grid:
        _loss, _acc = np.inf, 0
        if args.distributed:
            model = get_network(args.net, args.num_classes, bn_layer=nn.SyncBatchNorm).cuda(args.gpu)
            if args.eval:
                model.load_state_dict(torch.load(args.checkpt), strict=True)

            if args.lr_finder:
                if dist.get_rank() != 0:
                    return

                print("call the LRFinder !")
                model = get_network(args.net, args.num_classes, bn_layer=nn.BatchNorm2d).cuda(args.gpu)
                #import matplotlib.pyplot as plt
                optimizer = LARS(model.parameters(), lr=lr, weight_decay=args.wd) #, max_iters=max_iters)
                lr_finder = LRFinder(model, optimizer, criterion)
                lr_finder.range_test(train_loader, end_lr = 1, num_iter=100, step_mode='exp')
                #plt.ion()
                lr_finder.plot()

                return


            model = DDP(model, device_ids=[args.gpu])
        else:
            model = get_network(args.net, args.num_classes).cuda(args.gpu)


        if args.eval:
            loss, acc1, acc5 = test_one_epoch(logger, args, model, device, test_loader, writer, 0, criterion)

            if dist.get_rank() == 0:
                logger.info("Eval loss:{.6f}, acc1:{.4f}, acc5:{.4f}".format(loss, acc1, acc5))
            return

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif args.optimizer == 'lamb':
            optimizer = Lamb(model.parameters(), lr=lr, weight_decay=args.wd, betas=(.9, .999), adam=False)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = args.wd)
        elif args.optimizer == 'lars':
            optimizer = LARS(model.parameters(), lr=lr, weight_decay=args.wd) #, max_iters=max_iters)

        warmup_ratio = args.warmup * 1.0 / args.epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, \
                steps_per_epoch=len(train_loader), epochs=int(args.epochs*1.0), \
                anneal_strategy='linear', pct_start=warmup_ratio,
                div_factor=25, final_div_factor=10, cycle_momentum=False, \
                base_momentum=0.9, max_momentum=0.9)

        #scheduler = WarmUpLR(optimizer, len(train_loader) * args.warmup)
        #scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

        for epoch in range(1, args.epochs + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            #if epoch > args.warmup:
            #    scheduler2.step()

            train_loss_avg, train_acc1_avg, train_acc5_avg = train_one_epoch(logger, args, model, device, train_loader, optimizer, epoch, writer, criterion, scheduler)
            loss_tmp, acc_tmp, acc5_tmp = test_one_epoch(logger, args, model, device, test_loader, writer, epoch, criterion)

            if acc_tmp > _acc:
                _acc = acc_tmp
                _loss = loss_tmp

            if dist.get_rank() == 0:
                writer.add_scalars('loss_avg', {'train':train_loss_avg, 'test':loss_tmp}, epoch)
                writer.add_scalars('acc1_avg', {'train':train_acc1_avg, 'test':acc_tmp}, epoch)
                writer.add_scalars('acc5_avg', {'train':train_acc5_avg, 'test':acc5_tmp}, epoch)

        if op_acc < _acc:
            op_acc = _acc
            op_loss = _loss
            op_lr = lr
        if dist.get_rank() == 0:
            logger.info("LR:%.6f, test_loss:%.6f, test_acc:%.4f" % (lr, _loss, _acc))

    if dist.get_rank() == 0:
        logger.info("Optimal LR:%.6f, test_loss:%.6f, test_acc:%.4f" %(op_lr, op_loss, op_acc))
        #out = "Optimal LR:%s, batch-size:%s, test_loss:%s, test_acc:%s\n" % (op_lr, args.batch_size, op_loss, op_acc)
        #with open("optimizer_%s_%s_%s.txt"%(args.optimizer, args.dataset, args.batch_size), "w") as fout:
        #    fout.writelines(out)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

if __name__ == '__main__':
    main()
