import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import json
import math
from tqdm import tqdm
import numpy as np
import pickle
from IPython.display import clear_output
import albumentations as A
from datetime import datetime



def classnum(shape):
    if shape['label'] == 'fence':
        return 1
    if shape['label'] == 'forbidden':
        return 1
    if shape['label'] == 'building':
        return 1
    if shape['label'] == 'water':
        return 1
    return 1

def vec(x0, y0, x1, y1):
    return x0 * y1 - x1 * y0

def calc_vec(a, b, c):
    return vec(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

def intersec(a, b, c, d):
    return calc_vec(a, b, c) * calc_vec(a, b, d) < 0 and calc_vec(c, d, b) * calc_vec(c, d, a) < 0

def prepare(pic):
    res = pic / 255
    return res

def get_bounding_box(shape):
    if shape['shape_type'].lower() == 'line':
        v0, v1 = shape['points']
        return [min(v0[0], v1[0]), min(v0[1], v1[1])], [max(v0[0], v1[0]), max(v0[1], v1[1])]
    if shape['shape_type'].startswith('line'):
        bb = None
        for i in range(len(shape['points']) - 1):
            line = {'points': [shape['points'][i], shape['points'][i + 1]],
                    'shape_type': 'line'}
            b1 = get_bounding_box(line)
            if bb is None:
                bb = b1
            else:
                bb[0][0] = min(bb[0][0], b1[0][0])
                bb[0][1] = min(bb[0][1], b1[0][1])
                bb[1][0] = max(bb[1][0], b1[1][0])
                bb[1][1] = max(bb[1][1], b1[1][1])
        return bb
    if shape['shape_type'] == 'polygon':
        border = {
            'points': shape['points'],
            'shape_type': 'lines'
        }
        return get_bounding_box(border)
    if shape['shape_type'] == 'circle':
        v = shape['points'][0]
        v1 = shape['points'][1]
        r = np.linalg.norm([v1[0] - v[0], v1[1] - v[1]])
        return [v[0] - r, v[1] - r], [v[0] + r, v[1] + r]
    return None

def detect(x, y, shape):
    if shape['shape_type'] == 'Line':
        v0, v1 = shape['points']
        a = v1[1] - v0[1]
        b = v0[0] - v1[0]
        c = -(v1[0] * a + v1[1] * b)
        if a ** 2 + b ** 2 == 0:
            return False
        d = (a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
        between = ((x - v0[0]) * (v1[0] - v0[0]) + (y - v0[1]) * (v1[1] - v0[1]) ) >= 0 and ((x - v1[0]) * (v0[0] - v1[0]) + (y - v1[1]) * (v0[1] - v1[1]) ) >= 0
        return abs(d) < 3 and between
    if shape['shape_type'].startswith('line'):
        for i in range(len(shape['points']) - 1):
            line = {'points': [shape['points'][i], shape['points'][i + 1]],
                    'shape_type': 'Line'}
            if detect(x, y, line):
                return True
        return False
    if shape['shape_type'] == 'polygon':
        s = 0
        other = [10002.7, 10000.5] 
        for i in range(len(shape['points'])):
            point = shape['points'][i]
            next_point = shape['points'][(i + 1) % len(shape['points'])]
            if intersec([x, y], other, point, next_point):
                s += 1
        if s % 2 != 0:
            return True
        return False
    if shape['shape_type'] == 'circle':
        v = shape['points'][0]
        v1 = shape['points'][1]
        r = np.linalg.norm([v1[0] - v[0], v1[1] - v[1]])
        if abs(np.linalg.norm([x - v[0], y - v[1]]) - r) < 5:
            return True
    return False

class DraftDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.length = self.labels.shape[0]
        self.size = 100
        self.images = []
        self.annotations = []
        if os.path.isfile('data/X.data') and os.path.isfile('data/y.data'):
            self.X = pickle.load(open('data/X.data', 'rb'))
            self.y = pickle.load(open('data/y.data', 'rb'))
            return
        self.masks = []
        self.masks_processed = []
        for idx in tqdm(range(self.length)):
            assert idx < len(self.labels)
            img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
            print('Processing {0}'.format(img_path))
            pic = cv2.imread(img_path)
            b, g, r = cv2.split(pic) # по умолчанию cv2 почему-то отдает цвета в порядке BGR вместо RGB
            self.images.append(cv2.merge([r, g, b]))
            self.annotations.append(json.load(open(data_dir + self.labels.iloc[idx, 0].split('.')[0] + '.json', 'r')))
            img = self.images[-1]
            annotation = self.annotations[-1]
            img_shape = img.shape
            mask = np.zeros(shape=(img_shape[0], img_shape[1]))
            for shape in annotation['shapes']:
                lb, rt = get_bounding_box(shape)
                lb[0] = max(lb[0], 0)
                lb[1] = max(lb[1], 0)
                rt[0] = min(rt[0], img.shape[1] - 1)
                rt[1] = min(rt[1], img.shape[0] - 1)
                for i in range(int(lb[0] - 1), int(rt[0] + 1)):
                    for j in range(int(lb[1] - 1), int(rt[1] + 1)):
                        if detect(i, j, shape):
                            mask[j][i] = float(classnum(shape))
            self.masks.append(mask)
            print('Saving')
            clear_output(wait=True)
        self.X = self.images
        self.y = self.masks
        pickle.dump(self.y, open('data/y.data', 'wb'))
        pickle.dump(self.X, open('data/X.data', 'wb'))
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        img = self.X[idx]
        mask = self.y[idx]
        return img, mask




class AugmentedDataset(Dataset):
    def __init__(self, draft, augmentor, size = 250):
        self.draft = draft
        self.augmentor = augmentor
        self.size = size
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X, y = self.draft[idx]
        augmented = self.augmentor(image=X, mask=y)
        X = (np.transpose(augmented['image'], axes=[0, 1, 2]) / 255)
        y = np.array(augmented['mask'])
        return X, y

def get_data(size=128):
    dataset = DraftDataset('data/annotations.csv', 'data/')

    aug = A.Compose([
        A.RandomCrop(width=size, height=size, p=1),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(p=.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.3),  
    ])
    data = AugmentedDataset(dataset, aug)
    return data
