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

from optimizers import Lamb, log_lamb_rs
from optimizers import LARS



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch, event_writer):
    model.train()
    tqdm_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(tqdm_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            step = batch_idx * len(data) + (epoch-1) * len(train_loader.dataset)
            log_lamb_rs(optimizer, event_writer, step)
            event_writer.add_scalar('loss', loss.item(), step)
            lr = optimizer.param_groups[0]['lr']
            tqdm_bar.set_description(
                    f'Train epoch {epoch} Loss: {loss.item():.6f}, lr:{lr:.6f}')

def test(args, model, device, test_loader, event_writer:SummaryWriter, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    event_writer.add_scalar('loss/test_loss', test_loss, epoch - 1)
    event_writer.add_scalar('loss/test_acc', acc, epoch - 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc))

    return test_loss, correct

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--g', type=int, default=0, metavar='N',
                        help='device id (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'lars', 'adam', 'sgd'],
                        help='which optimizer to use')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=str, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                        help='weight decay (default: 0.01)')
    parser.add_argument('--seed', type=int, default=2020, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    print("args:", args)

    use_cuda = torch.cuda.is_available()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("fix the seed:", args.seed)
    device = torch.device("cuda:%s"%args.g if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    print("Use optimizer:",  args.optimizer)
    writer = SummaryWriter()

    N = 10
    lo, hi = args.lr.split(",")
    grid = np.linspace(float(lo), float(hi), N)
    print("search lr from:", lo, " to ", hi)
    op_loss, op_acc = 0, 0
    op_lr = 0
    for lr in grid:
        _loss, _acc = np.inf, 0
        model = Net().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif args.optimizer == 'lamb':
            optimizer = Lamb(model.parameters(), lr=lr, weight_decay=args.wd, betas=(.9, .999), adam=False)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = args.wd)
        elif args.optimizer == 'lars':
            max_iters = args.epochs * len(train_loader)
            print("set max_iters:", max_iters)
            optimizer = LARS(model.parameters(), lr=lr, weight_decay=args.wd, max_iters=max_iters)


        for epoch in range(1, args.epochs + 1):
            lr_ = (1-abs((epoch+1)/(args.epochs+1)*2-1))*lr
            optimizer.param_groups[0]['lr'] = lr_

            train(args, model, device, train_loader, optimizer, epoch, writer)
            loss_tmp, acc_tmp = test(args, model, device, test_loader, writer, epoch)
            if acc_tmp > _acc:
                _acc = acc_tmp
                _loss = loss_tmp
        if op_acc < _acc:
            op_acc = _acc
            op_loss = _loss
            op_lr = lr
        print("LR:", lr, " test_loss:", _loss, " test_acc:", _acc)

    print("Optimal LR:", op_lr, " test_loss:", op_loss, " test_acc:", op_acc)
    out = "Optimal LR:%s, batch-size:%s, test_loss:%s, test_acc:%s\n" % (op_lr, args.batch_size, op_loss, op_acc)
    with open("optimizer_%s_%s.txt"%(args.optimizer, args.batch_size), "w") as fout:
        fout.writelines(out)



if __name__ == '__main__':
    main()
