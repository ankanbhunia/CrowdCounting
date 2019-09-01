  
            
import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import scipy.io

import numbers

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
        #print img.size, mask.size
        #assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


def get_file_paths(path,ext):
    paths = np.array(sorted([os.path.join(root, file)  for root, dirs, files in os.walk(path)
                             for file in files if file.endswith(ext)]))
    return paths

class Load_dataset(data.Dataset):
    
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):

        self.img_all_paths = get_file_paths(data_path, 'png')
        self.gt_all_paths = get_file_paths(data_path, 'csv')
        #print len(self.img_all_paths)
        #print len(self.gt_all_paths)
        
        nodata = len(self.img_all_paths)
        
        if mode == 'train':
            self.img_all_paths = self.img_all_paths[:int(0.7*(nodata))]
            self.gt_all_paths = self.gt_all_paths[:int(0.7*(nodata))]
            #print(self.img_all_paths)
            #self.img_all_paths = get_file_paths('/home/bhuniaa/Project/WE/train/', 'jpg')
            #self.gt_all_paths = get_file_paths('/home/bhuniaa/Project/WE/train/', 'csv')
            
        elif mode == 'test':
            self.img_all_paths = self.img_all_paths[int(0.7*(nodata)):]
            self.gt_all_paths = self.gt_all_paths[int(0.7*(nodata)):]
            #print (self.img_all_paths)
            #self.img_all_paths = get_file_paths('/home/bhuniaa/Project/WE/test/', 'jpg')
            #self.gt_all_paths = get_file_paths('/home/bhuniaa/Project/WE/test/', 'csv')
        self.num_samples = len(self.img_all_paths) 
        
        self.mode = mode
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    
    def __getitem__(self, index):
        
        imgfname = self.img_all_paths[index]
        gtfname = self.gt_all_paths[index]
        
        input_size = []
        if self.mode=='train':
        	input_size = [720,576]
        elif self.mode=='test':
        	input_size =[720,576]

        # print fname
        img, den = self.read_image_and_gt(imgfname, gtfname, input_size)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 

        if self.img_transform is not None:
            img = self.img_transform(img)

        gt_count = torch.from_numpy(np.array(den)).sum() 

        if self.gt_transform is not None:
            den = self.gt_transform(den)      
            
        return img, den, gt_count

    
    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, imgfname, gtfname, input_size):
        img = Image.open(imgfname)
        if img.mode == 'L':
            img = img.convert('RGB')
        wd_1, ht_1 = img.size

        
        #den = scipy.io.loadmat(gtfname,verify_compressed_data_integrity=False)['im_density'][:,:,0]
        den = pd.read_csv(gtfname, sep=',',header=None).values
        #den = scipy.io.loadmat(gtfname,verify_compressed_data_integrity=False)['y']
        den = den.astype(np.float32, copy=False)
     
        den = Image.fromarray(den)
        

        



        return img, den
       

    def get_num_samples(self):
        return self.num_samples       
            
        