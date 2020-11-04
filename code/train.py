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

from dataset import Cityscapes
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
parser.add_argument("--lr", type=float, default=0.01, help="lr")
parser.add_argument("--model_type", type=str, default="fcn_resnet18", help="model")
parser.add_argument("--bs_train", type=int, default=6, help="bs for train")
parser.add_argument("--bs_val", type=int, default=4, help="bs for val")
parser.add_argument("--wd", type=float, default=1e-4, help="wd")
parser.add_argument("--epochs", type=int, default=400, help="epochs")
parser.add_argument("--resume", type=str, default=None, help="resume")
opts = parser.parse_args()

lr = opts.lr
model_type = opts.model_type
use_arma_layer = opts.use_arma
dataset_path = opts.datapath

model_save_folder = opts.save_folder + '/checkpoints/'
logs = opts.save_folder + '/logs/'

helper.make_dir(opts.save_folder)
helper.make_dir(logs)
helper.make_dir(model_save_folder)

bs_train = opts.bs_train
bs_val = opts.bs_val
wd = opts.wd
num_epochs = opts.epochs



writer = SummaryWriter(logs, flush_secs=10)

# ensure the experiment produces same result on each run
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)

# transforms
train_image_transform = torchvision.transforms.Compose([
  torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_target_transform = torchvision.transforms.Compose([
  torchvision.transforms.RandomHorizontalFlip(),
])

val_image_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# get cityscapes dataset from google drive link
train_dataset = Cityscapes(root=dataset_path, split='train', transform=train_image_transform,\
 target_transform=train_target_transform)

val_dataset = Cityscapes(root=dataset_path, split='val', transform=val_image_transform)

# get train and val loaders for the corresponding datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs_train, shuffle=True, \
  num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs_val, \
  shuffle=False, num_workers=16)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_map = {
    'fcn_resnet18': network.fcn_resnet18,
    'fcn_resnet50': network.fcn_resnet50,
    'deeplabv3_resnet18': network.deeplabv3_resnet18,
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101
}

model = model_map[model_type](arma=use_arma_layer)
for m in model.backbone.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.01

model = nn.DataParallel(model)
model.to(device)

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
scheduler = utils.PolyLR(optimizer, num_epochs, power=0.9)

# define loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

# train-eval loop
metrics = StreamSegMetrics(19)
train_loss_list = []
train_iou_list = []
val_loss_list = []
val_iou_list = []
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
  metrics.reset()
  model.train()
  train_loss = 0.0
  for step, (images, labels) in enumerate(train_loader):
    
    start = time.time()
    # if cuda
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    labels = labels.squeeze(1)
    
    # get loss
    optimizer.zero_grad()
    outputs = model(images)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # metrics
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()
    metrics.update(targets, preds)

    end = time.time()

    if step%10==0:
      print('Epoch: ',str(epoch),' Iter: ',step,'Loss: ',loss.item(),)

    print('iter time: ',end-start)  

  # update training_loss, training_accuracy and training_iou 
  train_loss = train_loss/float(len(train_loader))
  train_loss_list.append(train_loss)
  results = metrics.get_results()
  train_iou = results["Mean IoU"]
  train_iou_list.append(train_iou)

  
  writer.add_scalar("loss/train", train_loss,epoch)
  writer.add_scalar("iou/train", train_iou,epoch)


  if epoch%5==0:
    metrics.reset()
    model.eval()
    val_loss = 0.0
    for step, (images, labels) in enumerate(val_loader):
        with torch.no_grad():

            # if cuda
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels = labels.squeeze(1)

            # get loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # metrics
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

    # update val_loss, val_accuracy and val_iou 
    val_loss = val_loss / float(len(val_loader))
    val_loss_list.append(val_loss)
    results = metrics.get_results()
    val_iou = results["Mean IoU"]
    val_iou_list.append(val_iou)

    writer.add_scalar("loss/val", val_loss,epoch)
    writer.add_scalar("iou/val_iou", val_iou,epoch)

    state = {'state_dict': model.state_dict(),'epoch': epoch,\
    'optimizer':optimizer.state_dict(),'opts':opts}

    torch.save(state,model_save_folder+'/'+'current.pth')

    # store best model(early stopping)
    if(float(val_iou)>best_metric):
        best_metric = float(val_iou)
        best_metric_epoch = epoch
        #torch.save(model.state_dict(), model_save_folder)
        torch.save(state,model_save_folder+'/'+'model_best.pth')

    print()
    print("Epoch: " + str(epoch))
    print("Training Loss: " + str(train_loss) + "    Validation Loss: " + str(val_loss))
    print("Training mIoU: " + str(train_iou) + "    Validation mIoU: " + str(val_iou))
    print("Best Validation mIoU: " + str(best_metric))
    print()

  scheduler.step()
