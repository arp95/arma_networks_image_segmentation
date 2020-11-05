# main file: runs the training-eval loop for Deeplabv3/Deeplabv3+ on Cityscapes dataset
# header files
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
from random import shuffle
from PIL import Image
from collections import namedtuple
import json
import time 
from sklearn.metrics import confusion_matrix
import argparse

from metrics import StreamSegMetrics
import network 
import utils

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import helper

from torch.utils.tensorboard import SummaryWriter


# user-defined values
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default='./datasets/data', help="path to Dataset")
parser.add_argument("--save_folder", type=str, default='test/', help="path to save model")
parser.add_argument("--use_arma", type=bool, default=True, help="use arma layer or not")
parser.add_argument("--lr", type=float, default=0.001, help="lr")
parser.add_argument("--model_type", type=str, default="resnet18", help="model")
parser.add_argument("--bs_train", type=int, default=64, help="bs for train")
parser.add_argument("--bs_val", type=int, default=64, help="bs for val")
parser.add_argument("--wd", type=float, default=1e-4, help="wd")
parser.add_argument("--epochs", type=int, default=100, help="epochs")
parser.add_argument("--resume", type=str, default=None, help="resume")
opts = parser.parse_args()

lr = opts.lr
model_type = opts.model_type
use_arma_layer = opts.use_arma
dataset_path = opts.datapath
bs_train = opts.bs_train
bs_val = opts.bs_val
wd = opts.wd
num_epochs = opts.epochs

model_save_folder = opts.save_folder + '/checkpoints/'
logs = opts.save_folder + '/logs/'

helper.make_dir(opts.save_folder)
helper.make_dir(logs)
helper.make_dir(model_save_folder)
writer = SummaryWriter(logs, flush_secs=10)

# ensure the experiment produces same result on each run
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)

# transforms
train_transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop((224, 224)),
                                       torchvision.transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop((224, 224)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# get cityscapes dataset from google drive link
train_dataset = torchvision.datasets.ImageNet(root=dataset_path, train=True, transform=train_transforms, download=False)
val_dataset = torchvision.datasets.ImageNet(root=dataset_path, train=False, transform=val_transforms, download=False)

# get train and val loaders for the corresponding datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs_train, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs_val, shuffle=False, num_workers=16)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_map = {
    'resnet18': network.resnet18,
    'resnet50': network.resnet50,
    'resnet101': network.resnet101
}

model = model_map[model_type](arma=use_arma_layer)
model = nn.DataParallel(model)
model.to(device)

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

# define loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

# train-eval loop
train_losses = []
train_acc = []
val_losses = []
val_acc = []
best_metric = -1
best_metric_epoch = -1

print("Starting training: ")
print("Train images: ",len(train_dataset.images))
print("Val images: ",len(val_dataset.images))

start_epoch = 0
if opts.resume is not None:
  state = torch.load(opts.resume)
  start_epoch = state['epoch']+1
  model.load_state_dict(state['state_dict'])
  optimizer.load_state_dict(state['optimizer'])
  opts = state['opts']
  print("resuming")

for epoch in range(start_epoch, num_epochs):

  # train part
  model.train()
  training_loss = 0.0
  total = 0
  correct = 0
  for step, (images, target) in enumerate(train_loader):
    
    start = time.time()
    # if cuda
    images = images.to(device, dtype=torch.float32)
    target = target.to(device, dtype=torch.long)
    
    # get loss
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # metrics
    training_loss = training_loss + loss.item()
    _, predicted = output.max(1)
    total += target.size(0)
    correct += predicted.eq(target).sum().item()

    end = time.time()
    if step%10==0:
      print('Epoch: ', str(epoch), ' Iter: ', step, 'Loss: ', loss.item(),)
    print('iter time: ', end-start)  

  # update training_loss, training_accuracy and training_iou 
  training_loss = training_loss / float(len(train_loader))
  training_accuracy = str(100.0 * (float(correct) / float(total)))
  train_losses.append(training_loss)
  train_acc.append(training_accuracy)

  writer.add_scalar("loss/train", training_loss, epoch)
  writer.add_scalar("acc/train_acc", training_accuracy, epoch)


  if epoch%5==0:
    model.eval()
    valid_loss = 0.0
    total = 0
    correct = 0
    for step, (images, target) in enumerate(val_loader):
        with torch.no_grad():

            # if cuda
            images = images.to(device, dtype=torch.float32)
            target = labels.to(device, dtype=torch.long)

            output = model(input)
            loss = criterion(output, target)
            valid_loss = valid_loss + loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # update val_loss, val_accuracy and val_iou 
    valid_loss = valid_loss / float(len(val_loader))
    valid_accuracy = str(100.0 * (float(correct) / float(total)))
    val_losses.append(valid_loss)
    val_acc.append(valid_accuracy)

    writer.add_scalar("loss/val", valid_loss, epoch)
    writer.add_scalar("acc/val_acc", valid_accuracy, epoch)

    state = {'state_dict': model.state_dict(),'epoch': epoch,\
    'optimizer':optimizer.state_dict(),'opts':opts}
    torch.save(state,model_save_folder+'/'+'current.pth')

    # store best model(early stopping)
    if(float(valid_accuracy)>best_metric):
        best_metric = float(valid_accuracy)
        best_metric_epoch = epoch
        #torch.save(model.state_dict(), model_save_folder)
        torch.save(state,model_save_folder+'/'+'model_best.pth')

    print()
    print("Epoch" + str(epoch) + ":")
    print("Training Accuracy: " + str(training_accuracy) + "    Validation Accuracy: " + str(valid_accuracy))
    print("Training Loss: " + str(training_loss) + "    Validation Loss: " + str(valid_loss))
    print("Best Validation Accuracy: " + str(best_metric))
    print()
