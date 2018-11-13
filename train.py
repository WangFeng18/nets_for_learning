import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.cifar import CIFAR10
import torch.utils.data.dataloader as dl
import resnet
import os
import numpy as np
from tensorboardX import SummaryWriter
import time
import fire
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class config:
    max_epoch = 400
    lr = 0.1
    batch_size = 128
    net = 'resnet18'
    num_classes = 10

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

train_dataloader = dl.DataLoader(CIFAR10('data/', download=True, \
        transform=transform_train), batch_size=config.batch_size, shuffle=True, num_workers=4)
test_dataloader = dl.DataLoader(CIFAR10('data/', train=False, download=True, \
        transform=transform_test), batch_size=100, num_workers=4)
net = getattr(resnet, config.net)(pretrained=False, num_classes=config.num_classes)

def train(net, dataloader, config, load_model):
    # 1.define loss
    # 2.define opt
    if load_model: net.load_state_dict(torch.load(os.path.join('model', '{}.pth'.format(config.net))))
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # The initial lr will be decayed by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    writer = SummaryWriter()
    for i_epoch in range(config.max_epoch):
        scheduler.step()
        epoch_loss = 0.
        epoch_start = time.time()
        for i_batch, data in enumerate(dataloader):
            input = data[0].float().cuda()
            label = data[1].long().cuda()
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print('\tepoch: {:d}, batch: {:d}, loss:{:.4f}, lr:{:}'.format(i_epoch, i_batch, loss, scheduler.get_lr()))
            epoch_loss += loss
        epoch_end = time.time()
        print('epoch: {:d}, epoch_loss:{:.4f}, spent:{:.4f}'.format(i_epoch, epoch_loss/i_batch, epoch_end-epoch_start))
        writer.add_scalar('epoch_loss', epoch_loss/i_batch, i_epoch)
        writer.add_scalar('epoch_time', epoch_end-epoch_start, i_epoch)
        torch.save(net.state_dict(), os.path.join('model', '{}.pth'.format(config.net)))

def test(net, dataloader, load_model):
    if load_model:
        net.load_state_dict(torch.load(os.path.join('model', '{}.pth'.format(config.net))))
    net.cuda()
    n_correct = 0.
    n_total = 0.
    for i_batch, data in enumerate(dataloader):
        input = data[0].float().cuda()
        label = data[1].long().cuda()
        output = net(input)
        pred = output.argmax(1)
        n_correct += (pred == label).sum()
        n_total += data[0].size(0)
    print n_correct.cpu().numpy()/n_total



def firefunc(istrain=True, load_model=False):
    if istrain:train(net, train_dataloader, config, load_model)
    else:test(net, test_dataloader, True)
fire.Fire(firefunc)
