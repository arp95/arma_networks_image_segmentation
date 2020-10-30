# main file: runs the training-eval loop for Deeplabv3 on Cityscapes dataset
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
from sklearn.metrics import confusion_matrix

from dataset import Cityscapes
from metrics import StreamSegMetrics

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


# hyper-parameters
sgd_optim = True
batch_size_train = 8
batch_size_val = 4
lr = 1e-4
num_epochs = 1000

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
train_dataset = Cityscapes(root="/content/drive/My Drive/cityscapes/", split='train', transform=train_image_transform, target_transform=train_target_transform)
val_dataset = Cityscapes(root="/content/drive/My Drive/cityscapes/", split='val', transform=val_image_transform)

# get train and val loaders for the corresponding datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=16)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 19)
model.to(device)

# optimizer
if sgd_optim:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# define loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

# train-eval loop
metrics = StreamSegMetrics(19)
train_loss_list = []
train_accuracy_list = []
train_iou_list = []
val_loss_list = []
val_accuracy_list = []
val_iou_list = []
best_metric = -1
best_metric_epoch = -1

for epoch in range(0, num_epochs):

  # train part
  metrics.reset()
  model.train()
  train_loss = 0.0
  for step, (images, labels) in enumerate(train_loader):
    
    # if cuda
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    labels = labels.squeeze(1)
    
    # get loss
    optimizer.zero_grad()
    outputs = model(images)['out']

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # metrics
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()
    metrics.update(targets, preds)

  # update training_loss, training_accuracy and training_iou 
  train_loss = train_loss/float(len(train_loader))
  train_loss_list.append(train_loss)
  results = metrics.get_results()
  train_accuracy = results["Overall Acc"]
  train_iou = results["Mean IoU"]
  train_accuracy_list.append(train_accuracy)
  train_iou_list.append(train_iou)

  
  # eval part
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
      outputs = model(images)['out']
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
  val_accuracy = results["Overall Acc"]
  val_iou = results["Mean IoU"] 
  val_accuracy_list.append(val_accuracy)
  val_iou_list.append(val_iou)

  # store best model(early stopping)
  if(float(val_accuracy)>best_metric and epoch>=50):
    best_metric = float(val_accuracy)
    best_metric_epoch = epoch
    torch.save(model.state_dict(), "/content/drive/My Drive/deeplabv3_cityscapes.pth")

  print()
  print("Epoch: " + str(epoch))
  print("Training Loss: " + str(train_loss) + "    Validation Loss: " + str(val_loss))
  print("Training Accuracy: " + str(train_accuracy) + "    Validation Accuracy: " + str(val_accuracy))
  print("Training mIoU: " + str(train_iou) + "    Validhation mIoU: " + str(val_iou))
  print()
