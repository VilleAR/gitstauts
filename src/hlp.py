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



for i in range(1,10):
    print(i,'times',10-i)
    print(i*(10-i))

