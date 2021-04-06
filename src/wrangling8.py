import os as os
import shutil as shutil
import re as re 



directory=r'../test'
ims=[]
with open('gex2.txt') as f:
    lines=f.readlines()
    for l in lines:
        ims.append(str(l))
t=directory+'/'

for name in ims:
    target=t+name
    target=target.replace('\n','')
    os.remove(target)
