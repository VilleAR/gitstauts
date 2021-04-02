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
from PIL import Image 

#-----HYPERPARAMETERS
input_dim=16384 #128*128
hidden_dim=128
output_dim=14
NUM_CLASSES=14
BATCH_SIZE_TRAIN=1
BATCH_SIZE_VAL=1000
N_EPOCHS=5
USE_L1=False
USE_L2=False
LOG_INTERVAL=1e99
lambda1, lambda2=1e-6, 0.001

def wrangling():
    directory=r'images'
    imgs=[]
    for i in range(1,20001):
        im=directory+'/im'+str(i)+'.jpg'    
        trans=transforms.Compose([
                                            transforms.Resize(128),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor()                                     
        ])
        image=Image.open(im)
        image=image.convert('RGB')
        image=trans(image)
        #fp=np.asarray(image)
        imgs.append(image)

    arr2=[]
    with open('labels.txt') as f:
        lines=f.readlines()
        for l in lines:
            s=str(l)
            s=s[:-2]
            arr=s.split(' ')
            arr3 = [int(numeric_string) for numeric_string in arr]
            arr2.append(arr3)
    dataset=[]
    for i in range(0,20000):
        dataset.append((imgs[i],arr2[i]))

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.25)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


    #--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features = 12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=12 , out_channels=20 , kernel_size=3, stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20 , out_channels=32 , kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features= 32*64*64, out_features=NUM_CLASSES)
     

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(-1, 32*64*64)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.fc(x)
        return x #softmax OUT!!!!

    #--- set up --#

    #TRAINING!!!!!!!!!!!!!
def train(epoch):
    model.train()
    for batch_num, (data, target) in enumerate(train_loader):       
        data, target = data.to(device), target.to(device)        
        output = model(data)
        loss = loss_function(output, target)
        train_losses.append(loss.item())
        train_counter.append((batch_num*100)+((epoch-1)*len(train_loader.dataset)))
        l1_reg = 0.0
        l2_reg = 0.0            
        optimizer.zero_grad()
        '''
        for p in model.parameters(): 
            l1_reg+=lambda1*torch.norm(p,1)
            l2_reg+= lambda2 * torch.norm(p, 2)**2
        '''
        if USE_L1:
            loss+=l1_reg
        if USE_L2:
            loss+=l2_reg     
        loss.backward()
        optimizer.step()
        # WRITE CODE HERE   
        if batch_num%LOG_INTERVAL==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_num * len(data), len(train_loader.dataset),
                100. * batch_num / len(train_loader), loss.item()))

#VAlidation
def validate():
    model.eval()
    val_loss=0
    correct=0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target=data.to(device), target.to(device)
            output=model(data)
            val_loss+=loss_function(output, target).item()
            pred=output.data.max(1, keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
        val_loss/=len(val_loader.dataset)
        val_losses.append(val_loss)
        acc=correct/len(val_loader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return acc
if __name__ == '__main__':  
    #------DATA WRANGLING
    train_transform = transforms.Compose([
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])
    val_transform=transforms.Compose([transforms.ToTensor()])
    #dataset = ImageFolder('../data/alldata', transform=train_transform)
    datasets = wrangling()

    train_loader = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader=torch.utils.data.DataLoader(dataset=datasets['val'], batch_size=BATCH_SIZE_VAL, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = CNN().to(device)

    #Optimizers!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!????????????!!!!!!!!!!!!!!!!!
    optimizer=optim.Adam(model.parameters())
    loss_function=nn.BCELoss()
    #Train in main!!!!!!!!
    prev=0
    count=0
    bestac=0
    bestepoch=0
    acccs=[]
    for e in range(1, N_EPOCHS+1):
        train_losses = []
        train_counter = []
        val_losses=[]
        train(e)
        #acc=validate()   
        #acccs.append(acc)
