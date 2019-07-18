import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision as tv
import pandas as pd

#import nltk
from PIL import Image
import os
import pickle
import torchvision.transforms as transforms


def set_transform(resize=(256,256), crop_size=(224,224), horizontal_flip=False, normalize=True):
    compose_lst = []
    if resize is not None: compose_lst.append(transforms.Resize(resize))
    if crop_size is not None: compose_lst.append(transforms.RandomCrop(crop_size))
    if horizontal_flip: compose_lst.append(transforms.RandomHorizontalFlip())
    compose_lst.append(transforms.ToTensor())
    if normalize: compose_lst.append(transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)))

    transform = transforms.Compose(compose_lst)
    
    return transform
    
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None,loader=tv.datasets.folder.default_loader):
        self.df = df
        self.transform=transform
        self.images= self.df['images']
        self.classes= self.df['classes']
        

    def __getitem__(self, index):
        img_id = self.images[index]
        
        image = Image.open(img_id).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        classes = self.classes[index]
        
        
        return image, classes


    def __len__(self):
        n, _ = self.df.shape
        return n
    

def get_loader( csv_file,batch_size, num_workers=4,transform=None,shuffle=False,):

    df = pd.read_csv(csv_file)
    train_dataset = ImagesDataset(df,transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)

    return train_loader
