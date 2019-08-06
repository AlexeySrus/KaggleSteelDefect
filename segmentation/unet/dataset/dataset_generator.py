import cv2
import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from torchvision.transforms import ToTensor, Normalize


def rleToMask(rleString, height, width):
    rows, cols = height, width
    if rleString == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
            img = img.reshape(cols,rows)
            img = img.T
        return img


class SteelDatasetGenerator(Dataset):
    def __init__(self, dataset_path, table_path, validation=False, validation_part=0.2):
        table_data = pd.read_csv(table_path).fillna(-1).values

        self.channels_data = {}
        for img_id_class, data in table_data:
            img_id, channel_class = img_id_class.split('_')

            if img_id not in self.channels_data.keys():
                self.channels_data[img_id] = {}

            self.channels_data[img_id][int(channel_class)] = data

        self.images_names_list = list(set(map(lambda x: x.split('_')[0], table_data[:, 0])))
        self.dataset_path = dataset_path

        self.images_names_list = \
            self.images_names_list[:-int(len(self.images_names_list) * validation_part)] \
            if validation else \
            self.images_names_list[-int(len(self.images_names_list) * validation_part):]

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        image = np.array(Image.open(
            os.path.join(
                self.dataset_path,
                self.images_names_list[idx]
            )
        ).convert('LA'))[..., 0].astype(np.float32) / 255.0

        channels = np.array([
            rleToMask(
                self.channels_data[self.images_names_list[idx]][i],
                image.shape[0],
                image.shape[1]
            )
            for i in range(1, 5)
        ]).astype(np.float32) / 255.0

        return torch.FloatTensor(image).unsqueeze(0), torch.FloatTensor(channels)
