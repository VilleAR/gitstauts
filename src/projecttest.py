import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import Datakiller
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import random
import pandas as pd 
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from PIL import Image 

def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-12
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    acc=(tp+tn)/(tp+tn+fp+fn)
    return f1, acc

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, padding=1, kernel_size=KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(num_features = 12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=12 , out_channels=20 , kernel_size=KERNEL_SIZE, stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=12 , out_channels=32 , kernel_size=KERNEL_SIZE, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features= 32*64*64, out_features=NUM_CLASSES)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = F.dropout(x)
        #x = self.conv2(x)
        #x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = x.view(-1, 32*64*64)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.fc(x)
        #x = self.sig(x)
        return x 


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model=CNN().to_device()
cp=torch.load("checkpoint.pt")
model.load_state_dict(cp)
