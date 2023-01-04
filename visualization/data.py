import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import re

def dir_to_img_tensor(*paths):
    #return batch_size x channel x width x height
    layers_in_station = {}
    for path in paths:
        if not path.endswith('/'):
            path += '/'
        files = os.listdir(path)
        for file in files:
            file = path+file
            expression = r"station_(?P<label>\w+)_(?P<station_id>\d+)_(?P<zoom>\d+)_(?P<mode>\w+).png"
            regex = re.compile(expression)
            label,station_id,zoom,mode = regex.findall(file)[0]
            
            if label=="true":
                label = 1.
            elif label=="false":
                label = 0.
            
            img = cv2.imread(file)
            #img = np.transpose(img, (2,0,1))

            if not (station_id, zoom) in layers_in_station.keys():
                station_data = {'X':[img],'y':label}
                layers_in_station[(station_id, zoom)] = station_data
            else:
                layers_in_station[(station_id, zoom)]['X'].append(img) #image file은 특정 순서대로 저장됨

    X = np.array([np.concatenate(station_info['X'],axis=2) for station_info in layers_in_station.values()])
    y = np.array([station_info['y'] for station_info in layers_in_station.values()]).reshape([-1,1])

    return X, y

def compose_transform(num_channel, img_size = (448,448)):
    assert len(img_size) == 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*num_channel, [0.5]*num_channel),
        transforms.Resize(img_size)
    ])
    return transform

class MapDataset(Dataset):
    def __init__(self, img, label, transform=None):
        self.X = img
        self.y = label
        self.transform = transform

        if len(self.y.shape) == 1:
            self.y = self.y.reshape([-1,1])
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_x = self.X[idx]
        sample_y = self.y[idx]

        if self.transform:
            sample_x = self.transform(sample_x)

        return sample_x, sample_y

def save_dataset(dataset, file_name):
    torch.save(dataset, file_name)

def load_dataset(file_name):
    return torch.load(file_name)

