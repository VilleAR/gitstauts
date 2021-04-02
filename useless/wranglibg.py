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