import os

import PIL
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_data(img_path, dataset='UCSD'):
    if dataset == 'UCSD' or dataset == 'Mall':
        gt_path = img_path.replace('.jpg', '.csv').replace('frames_backup', 'csvs')
    else:
        gt_path = img_path.replace('.jpg', '.csv')
    img = Image.open(img_path).convert('RGB')
    target = np.loadtxt(gt_path, delimiter=',')

    if dataset == 'UCSD':
        img = img.resize((952, 632), resample=PIL.Image.BILINEAR)

    # resizing target based on the dataset
    if dataset == 'UCSD':
        target = cv2.resize(target, (target.shape[1] / 2, target.shape[0] / 2), interpolation=cv2.INTER_LINEAR) * 4
    else:
        # print("Not UCSD")
        target = cv2.resize(target, (target.shape[1] , target.shape[0]), interpolation=cv2.INTER_CUBIC) 
    return img, target


class TestWEDataset(Dataset):
    def __init__(self, dataset):

        self.root = '/home/bhuniaa/Project/WE10/train/'
        self.images = self.load_images()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
        self.dataset = dataset

    def load_images(self):
        folders = [os.path.join(self.root, img) for img in os.listdir(self.root)]
        images = []
        for folder in folders:
            imgs = [os.path.join(folder, img) for img in os.listdir(folder) if 'jpg' in img]
            images += imgs
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]

        img, target = load_data(img_path, self.dataset)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
