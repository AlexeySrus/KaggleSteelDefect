import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def rle2mask(rle, height, width):
    mask = np.zeros(width * height).astype(np.uint8)

    if rle == -1:
        return mask.reshape(width, height).T

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


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

        h, w = image.shape[:2]

        choose_x = np.random.randint(0, w - h + 1)

        channels = np.array([
            rle2mask(
                self.channels_data[self.images_names_list[idx]][i],
                image.shape[0],
                image.shape[1]
            )[:, choose_x:choose_x + h]
            for i in range(1, 5)
        ]).astype(np.float32) / 255.0

        image = image[:, choose_x:choose_x + h]

        return torch.FloatTensor(image).unsqueeze(0), torch.FloatTensor(channels)
