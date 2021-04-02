import os 

directory=r'annotations'
i=1
tensors=[]
log_interval=1000
for name in os.listdir(directory):
    s=directory+'/'+name
    with open(s) as f:
        lines=f.readlines()
        ints=[]
        for b in lines:
            ints.append(int(b))
        ints.sort()
    print(name)
    print(ints[0:10])

