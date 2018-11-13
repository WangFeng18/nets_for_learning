from activation import analyzer
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dl
import resnet
import os

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
activater = analyzer.Analyzer(net, test_dataloader)
activater.forward()
