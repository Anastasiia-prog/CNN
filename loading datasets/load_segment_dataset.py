from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms as T
import torch
from datasets_classes_names import import_classes


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch.nn as nn
import torchvision

from PIL import Image
import cv2
import albumentations as A

import os


class Segmentation_dataset:
    def __init__(self, batch_size=8, dataset_name='ADEChallengeData2016', image_resize=256):
        
        self.dataset_name = dataset_name
        
        t_train = A.Compose([A.Resize(image_resize, image_resize, interpolation=cv2.INTER_NEAREST), 
                             A.HorizontalFlip(), A.VerticalFlip(), 
                             A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                             A.GaussNoise()])

        t_val = A.Compose([A.Resize(image_resize, image_resize, interpolation=cv2.INTER_NEAREST), 
                           A.HorizontalFlip(),
                           A.GridDistortion(p=0.2)])
        
        images = Get_images(dataset_name=self.dataset_name)
        IMAGE_PATH_TRAIN, IMAGE_PATH_VAL, MASK_PATH_TRAIN, MASK_PATH_VAL = images.get_paths()
        X_train, X_val, X_test = images.split_data()
        ###
        X_train = X_train[:len(X_train)-1]
        ###
        # datasets
        self.train_set = DroneDataset(IMAGE_PATH_TRAIN, MASK_PATH_TRAIN, X_train, t_train,)
        self.val_set = DroneDataset(IMAGE_PATH_VAL, MASK_PATH_VAL, X_val, t_val)
        self.test_set = DroneDataset(IMAGE_PATH_VAL, MASK_PATH_VAL, X_test, t_val)

        # dataloader
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=2) 
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False) 
    
    def get_datasets_sizes(self):
        print(f'Train size: {len(self.train_set)}',
              f'Validation size: {len(self.val_set)}',
              f'Test size: {len(self.test_set)}', sep='\n')
        
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_test_dataset(self):
        return self.test_set, self.train_set


class Get_images:
    def __init__(self, dataset_name):
        self.IMAGE_PATH_TRAIN = dataset_name + '/images/train/training/'
        self.IMAGE_PATH_VAL = dataset_name + '/images/val/validation/'
        self.MASK_PATH_TRAIN = dataset_name + '/annotations/train/training/'
        self.MASK_PATH_VAL = dataset_name + '/annotations/val/validation/'

    def get_paths(self):
        return self.IMAGE_PATH_TRAIN, self.IMAGE_PATH_VAL, self.MASK_PATH_TRAIN, self.MASK_PATH_VAL
    
    def create_df(self, IMAGE_PATH):
        name = []
        for dirname, _, filenames in os.walk(IMAGE_PATH):
            for filename in filenames:
                name.append(filename.split('.')[0])

        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))
    
    def split_data(self):
        train = self.create_df(self.IMAGE_PATH_TRAIN)
        val = self.create_df(self.IMAGE_PATH_VAL)
        
        X_train = train['id']
        X_val, X_test = train_test_split(val['id'].values, test_size=0.5)
        
        return X_train, X_val, X_test


class DroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if isinstance(img, np.ndarray):
            if self.transform is not None:
                aug = self.transform(image=img, mask=mask)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']

            if self.transform is None:
                img = Image.fromarray(img)

            t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img = t(img)
            mask = torch.from_numpy(mask).long()

            return img, mask

