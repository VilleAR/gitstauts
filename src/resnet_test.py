import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision import transforms, datasets
import torchvision.models as models
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
from sklearn.metrics import f1_score

from PIL import Image 

###Testing with pretrained Resnet32


minirun = False

#-----HYPERPARAMETERS
input_dim=16384
hidden_dim=128
if minirun:
    input_dim=4096 #128*128
    hidden_dim=64

output_dim=14
NUM_CLASSES=14
BATCH_SIZE=300
N_EPOCHS=15
USE_L1=False
USE_L2=False
LOG_INTERVAL=5
lambda1, lambda2=1e-5, 0.001
DATA_PATH = '../'
TRAIN_DATA = 'train'
VAL_DATA = 'test'
TRAIN_IMG_FILE = 'imstrain.txt'
VAL_IMG_FILE = 'imsval.txt'
TRAIN_LABEL_FILE = 'labelstrain.txt'
VAL_LABEL_FILE = 'labelsval.txt'
KERNEL_SIZE=3


def wrangling():
    trans_train=transforms.Compose([    
                                        
                                        
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(25), # -25 to 25 degree rotation
                                        transforms.ToTensor(), # scales the pixel values to the range [0,1]
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_val = transforms.Compose([
                                        
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(), # scales the pixel values to the range [0,1]
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
                 
    dset_train=Datakiller.Datakiller(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, trans_train)
    dset_val = Datakiller.Datakiller(
    DATA_PATH, VAL_DATA, VAL_IMG_FILE, VAL_LABEL_FILE, trans_val)

    return dset_train, dset_val

def f1_home(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:

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


def loss_for_f1(y_pred, y_true):

    y_pred = F.softmax(y_pred, dim = 1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-12

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return 1 - torch.mean(f1)



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
        #x = F.dropout(x)
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



    #--- set up --#

    #TRAINING!!!!!!!!!!!!!
def train(epoch):
    
    model.train()
    
    for batch_num, (data, target) in enumerate(train_loader):       
        data, target = data.to(device), target.to(device)        
        output = model(data)
        loss = loss_function(output, target.float())
        #loss=loss_for_f1(output, target.float())
        train_losses.append(loss.item())
        #train_losses.append(loss)
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

    z=5
    with torch.no_grad():
        epoch_total_f1=0
        epoch_total_acc=0
        epoch_skf1=0
        skf1s=[]
        ewf1=0
        for data, target in val_loader:
            groups=len(val_loader)           
            data, target=data.to(device), target.to(device)
            output=model(data)
            loss=loss_function(output, target.float())
            val_loss+=loss
            correct=0
            #print(target)
            i=0
            f1=0
            skf1=0
            sig=nn.Sigmoid()
            f1scoreout=sig(output)
            l=len(output)
            acc=0
            while i<l:
                f1l, a= f1_home(f1scoreout[i],target[i])         
                f1+=f1l
                sk=f1_score(target[i].cpu(), f1scoreout[i].round().cpu(), average='weighted')
                skf1+=sk
                acc+=a
                i-=-1  
            #print("F1 score:", f1/l)
            epoch_total_f1+=(f1/l)
            epoch_skf1+=(skf1/l)
            epoch_total_acc+=(acc/l)
        val_loss/=len(val_loader.dataset)
        val_losses.append(val_loss)
        acc=correct/len(val_loader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / 2493))
        print("Average Weighted F1 for the epoch: ", epoch_skf1/groups)
        print("Average home-made F1 for the epoch: ", epoch_total_f1/groups)
        print("Average accuracy for the epoch: ", epoch_total_acc/groups)
        print("**************************************************")
        print("\n")
   
        return (epoch_total_f1/groups), (epoch_total_acc/groups), (skf1/groups)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':  
    #------DATA WRANGLING

    #dataset = ImageFolder('../data/alldata', transform=train_transform)
    dstrain, dsval = wrangling()
    train_loader = torch.utils.data.DataLoader(dataset=dstrain, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
    val_loader =torch.utils.data.DataLoader(dataset=dsval, batch_size=BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    #Testing RESNET
    model_r = models.resnet34(pretrained = True)
    feature_count = model_r.fc.in_features
    model_r.fc = nn.Linear(feature_count, 14)
    model = model_r.to(device)

    #Optimizers!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!????????????!!!!!!!!!!!!!!!!!
    #optimizer=optim.Adadelta(model.parameters())
    optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    pos_weight=torch.ones([14])
    trainsize=7683
    sizes=[95, 360, 319, 1095, 448, 3227, 761, 2979, 598, 6403, 3121, 120, 173, 525]
    news=[]
    for s in sizes:
        news.append(s*0.7683)
    pos_weights=torch.ones([14])
    for a, i in enumerate(pos_weights):
        pos_weights[a]=i*(trainsize-news[a])/news[a]
    loss_function=nn.BCEWithLogitsLoss()
    #loss_function=nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    #loss_function = loss_for_f1
    #Train in main!!!!!!!!
    prev=0
    count=0
    bestac=0
    bestf=0
    bestepoch=0
    acccs=[]
    f1s=[]
    sks=[]
    for e in range(1, N_EPOCHS+1):
        train_losses = []
        train_counter = []
        val_losses=[]
        train(e)
        acc, f1, sk=validate()
        acccs.append(acc)
        f1s.append(f1)
        sks.append(sk)
        if f1>bestf:
            torch.save(model.state_dict(), "checkpoint2sgd.pt")
            bestf=f1
            bestac=acc
            bestepoch=e
    
    print("Best epoch at epoch: ", bestepoch)
    print("F1 at best epoch: ",bestf, "Accuracy at best epoch: ", bestac)
    cp=torch.load("checkpoint2sgd.pt")
    model.load_state_dict(cp)
    validate()
    plt.xlabel('Epoch')
    plt.plot(acccs)
    plt.plot(f1s)
    plt.plot(sks)
    plt.ylabel('Weighted and non-weighted F1Score and accuracies')
    plt.xlabel('Epoch')
    plt.savefig("Accuracies")