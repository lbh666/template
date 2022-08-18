from json import load
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from torchvision import datasets, transforms
import torch
import torch.utils.data as data_utl
from sklearn import preprocessing
import pandas as pd
# import h5py as h5
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader

class my_dataset(data_utl.Dataset):
    def __init__(self, cfg, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]), **kwargs):
        super().__init__()
        self.cfg = cfg
        self.dataset = cfg.get('dataset', 'DVSGait')
        self.size = cfg.get('size', None)
        self.root = cfg.get('root', None)
        self.filname_list = sorted(os.listdir(os.path.join(self.root, 'data')))
        self.transform = transform
    
    def __getitem__(self, index):
        file_name = self.filname_list[index]
        img = cv.imread(os.path.join(self.root, 'data', file_name), cv.IMREAD_GRAYSCALE)
        label = torch.zeros(9)
        label[ int(file_name.split('.')[0])-1 ] += 1
        if self.transform :
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.filname_list)


if __name__ == '__main__':
    cfg = {
        'dataset': 'default',
        'size': (28,28),
        'root': r'C:\Users\LBH666\Desktop\Digital-handwriting-recognition\Dataset\train'
    }
    dataset = my_dataset(cfg)
    loader = DataLoader(dataset, 
                            batch_size=2,
                            shuffle=True)
    img, label = next(iter(loader))
    print(img.shape)
    print(label)
    
