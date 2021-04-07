import numpy
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


trainsize=7683
sizes=[95, 360, 319, 1095, 448, 3227, 761, 2979, 598, 6403, 3121, 120, 173, 525]
news=[]
for s in sizes:
    news.append(s*0.7683)
pos_weights=torch.ones([14])
for a, i in enumerate(pos_weights):
    pos_weights[a]=i*(trainsize-news[a])/news[a]
    
print(pos_weights)
