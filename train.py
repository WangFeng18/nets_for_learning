import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.datasets.cifar import CIFAR10
import torch.utils.data.dataloader as dl
import resnet
import os
import numpy as np
from tensorboardX import SummaryWriter
import time
import fire
import torchvision.transforms as transforms
from utils import progress_bar
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class config:
    max_epoch = 400
    lr = 0.001
    batch_size = 512 
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
net = nn.DataParallel(net)
cudnn.benchmark = True

def train(net, dataloader, config, load_model):
    # 1.define loss
    # 2.define opt
    if load_model: net.load_state_dict(torch.load(os.path.join('model', '{}.pth'.format(config.net))))
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # The initial lr will be decayed by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    writer = SummaryWriter()
    for i_epoch in range(config.max_epoch):
        scheduler.step()
        epoch_loss = 0.
        epoch_start = time.time()
        n_total = 0.
        n_correct = 0.
        print('Epoch:{}'.format(i_epoch))
        for i_batch, data in enumerate(dataloader):
            input = data[0].float().cuda()
            label = data[1].long().cuda()
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            n_total += label.size(0)
            n_correct += predicted.eq(label).sum().item()

            loss.backward()
            optimizer.step()
            progress_bar(i_batch, len(dataloader), 'L:{:.4f}|A:{:.3f}% ({}/{})'.\
                    format(epoch_loss/(i_batch+1), n_correct/n_total*100., \
                    int(n_correct), int(n_total)))
        epoch_end = time.time()
        writer.add_scalar('epoch_loss', epoch_loss/i_batch, i_epoch)
        writer.add_scalar('epoch_time', epoch_end-epoch_start, i_epoch)
        test(net, test_dataloader, False)
        torch.save(net.state_dict(), os.path.join('model', '{}.pth'.format(config.net)))

def test(net, dataloader, load_model):
    if load_model:
        net.load_state_dict(torch.load(os.path.join('model', '{}.pth'.format(config.net))))
    net.cuda()
    net.eval()
    n_correct = 0.
    n_total = 0.
    epoch_loss = 0.
    criterion = nn.CrossEntropyLoss()
    for i_batch, data in enumerate(dataloader):
        input = data[0].float().cuda()
        label = data[1].long().cuda()
        output = net(input)
        loss = criterion(output, label)

        epoch_loss += loss.item()
        _, predicted = output.max(1)
        n_total += label.size(0)
        n_correct += predicted.eq(label).sum().item()

        progress_bar(i_batch, len(dataloader), 'L:{:.4f}|A:{:.3f}% ({}/{})'.format(epoch_loss/(i_batch+1), n_correct/n_total*100., int(n_correct), int(n_total)))
    net.train()



def firefunc(istrain=True, load_model=False):
    if istrain:train(net, train_dataloader, config, load_model)
    else:test(net, test_dataloader, True)
fire.Fire(firefunc)
