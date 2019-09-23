import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapOnImage


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


class SegmentationTrainTransform(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self.norm = Normalize(mean=mean, std=std)

        gaussian_blur_sigma_max = 1.0
        gaussian_noise_sigma_max = 0.05

        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Sequential(
                            children=[
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                    rotate=(-5, 5),
                                    shear=(-16, 16),
                                    order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                    mode="reflect",
                                    name="Affine"),
                                iaa.PerspectiveTransform(
                                    scale=0.0,
                                    name="PerspectiveTransform"),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.PiecewiseAffine(
                                        scale=(0.0, 0.01),
                                        nb_rows=(4, 20),
                                        nb_cols=(4, 20),
                                        order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                        mode="reflect",
                                        name="PiecewiseAffine"))],
                            random_order=True,
                            name="GeomTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.75,
                                    then_list=iaa.Add(
                                        value=(-10, 10),
                                        per_channel=0.5,
                                        name="Brightness")),
                                # iaa.Sometimes(
                                #     p=0.05,
                                #     then_list=iaa.Emboss(
                                #         alpha=(0.0, 0.5),
                                #         strength=(0.5, 1.2),
                                #         name="Emboss")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.Sharpen(
                                        alpha=(0.0, 0.5),
                                        lightness=(0.5, 1.2),
                                        name="Sharpen")),
                                iaa.Sometimes(
                                    p=0.25,
                                    then_list=iaa.ContrastNormalization(
                                        alpha=(0.5, 1.5),
                                        per_channel=0.5,
                                        name="ContrastNormalization"))
                            ],
                            random_order=True,
                            name="ColorTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 255.0 * 2.0 * gaussian_noise_sigma_max),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.001),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise")
                    ],
                    random_order=True,
                    name="MainProcess")])

    def __call__(self, src_img, src_masks):
        seq_det = self.seq.to_deterministic()

        segmaps = [
            SegmentationMapOnImage(
                src_mask // 255,
                shape=src_img.shape,
                nb_classes=2
            ) for src_mask in src_masks
        ]

        used_segmentation_to_masks = [
            src_mask.sum() > 0
            for src_mask in src_masks
        ]

        img_and_masks = [
            seq_det(image=src_img, segmentation_maps=segmap)
            if i == 0 or used_segmentation_to_masks[i] else
            (None, src_masks[i])
            for i, segmap in enumerate(segmaps)
        ]

        img = img_and_masks[0][0]

        masks = [mask[1] for mask in img_and_masks]

        masks = np.array([
            mask.draw(size=src_img.shape[:2])[..., 0].astype(np.uint8)
            if used_segmentation_to_masks[i] else
            src_masks[i]
            for i, mask in enumerate(masks)
        ])

        masks_elements = masks >= 100

        masks[masks_elements] = 255
        masks[np.bitwise_not(masks_elements)] = 0

        return img, masks


class SteelDatasetGenerator(Dataset):
    def __init__(self, dataset_path, table_path,
                 validation=False, validation_part=0.2,
                 augmentation=False):
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
            self.images_names_list[-int(len(self.images_names_list) * validation_part):] \
            if validation else \
            self.images_names_list[:-int(len(self.images_names_list) * validation_part)]

        self.augmentation = None
        if augmentation:
            self.augmentation = SegmentationTrainTransform()

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        image = np.array(Image.open(
            os.path.join(
                self.dataset_path,
                self.images_names_list[idx]
            )
        ).convert('LA'))[..., 0]

        h, w = image.shape[:2]

        choose_x = np.random.randint(0, w - h + 1)

        channels = [
            rle2mask(
                self.channels_data[self.images_names_list[idx]][i],
                image.shape[0],
                image.shape[1]
            )
            for i in range(1, 5)
        ]

        if self.augmentation is not None:
            image, channels = self.augmentation(
                image,
                channels
            )

        channels = [
            ch[:, choose_x:choose_x + h].astype(np.float32) / 255.0
            for ch in channels
        ]
        image = image[:, choose_x:choose_x + h].astype(np.float32) / 255.0

        return torch.FloatTensor(image).unsqueeze(0), \
               torch.FloatTensor(channels)


class OneClassSteelDatasetGenerator(Dataset):
    def __init__(self, dataset_path, table_path,
                 class_index, part_without_masks_relatively_with_masks=0.5,
                 validation=False, validation_part=0.2,
                 augmentation=False, preused_table=None):
        assert 1 <= class_index <= 4

        self.class_index = class_index

        if preused_table is None:
            table_data = pd.read_csv(table_path).fillna(-1)
            select_class_table = \
                table_data[[
                    '_{}'.format(class_index) in name
                    for name in table_data.loc[:, 'ImageId_ClassId'].values
                ]]

            selected_class_mask_with_data = select_class_table[
                                                'EncodedPixels'] != -1
            selected_class_mask_without_data = select_class_table[
                                                   'EncodedPixels'] == -1

            table_with_masks = select_class_table[selected_class_mask_with_data]
            table_without_masks = select_class_table[
                selected_class_mask_without_data].sample(
                int(
                    len(
                        table_with_masks
                    ) * part_without_masks_relatively_with_masks
                )
            )

            table_data = pd.concat(
                [table_with_masks, table_without_masks]
            ).sample(frac=1).values
        else:
            table_data = preused_table

        self.fixed_table_data = table_data

        self.channels_data = {}
        for img_id_class, data in table_data:
            img_id, channel_class = img_id_class.split('_')

            if img_id not in self.channels_data.keys():
                self.channels_data[img_id] = {}

            self.channels_data[img_id][int(channel_class)] = data

        self.images_names_list = list(
            set(map(lambda x: x.split('_')[0], table_data[:, 0]))
        )
        self.dataset_path = dataset_path

        self.images_names_list = \
            self.images_names_list[
                -int(len(self.images_names_list) * validation_part):] \
            if validation else \
            self.images_names_list[
                :-int(len(self.images_names_list) * validation_part)]

        self.augmentation = None
        if augmentation:
            self.augmentation = SegmentationTrainTransform()

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        image = np.array(Image.open(
            os.path.join(
                self.dataset_path,
                self.images_names_list[idx]
            )
        ).convert('LA'))[..., 0]

        h, w = image.shape[:2]

        choose_x = np.random.randint(0, w - h + 1)

        channels = [
            rle2mask(
                self.channels_data[self.images_names_list[idx]][
                    self.class_index],
                image.shape[0],
                image.shape[1]
            )
        ]

        if self.augmentation is not None:
            image, channels = self.augmentation(
                image,
                channels
            )

        channels = [
            ch[:, choose_x:choose_x + h].astype(np.float32) / 255.0
            for ch in channels
        ]
        image = image[:, choose_x:choose_x + h].astype(np.float32) / 255.0

        return torch.FloatTensor(image).unsqueeze(0), \
               torch.FloatTensor(channels)[0].unsqueeze(0)


class OneClassSteelTestDatasetGenerator(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_names_list = os.listdir(self.dataset_path)
        self.images_names_list = [fname for fname in self.images_names_list if fname.endswith('.jpg')]
        self.hh = 00

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.dataset_path,
            self.images_names_list[idx]
        )

        image = np.array(Image.open(img_path).convert('LA'))[..., 0]
        image = image.astype(np.float32) / 255.0

        return self.images_names_list[idx], torch.FloatTensor(image).unsqueeze(0)
