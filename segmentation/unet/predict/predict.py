import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from segmentation.unet.utils.image_utils import rle2str, mask2rle, split_on_tiles_h, combine_tiles
from segmentation.unet.model.model import Model
from segmentation.unet.dataset.dataset_generator import\
    OneClassSteelDatasetGenerator, OneClassSteelTestDatasetGenerator
from torch.utils.data import DataLoader
from segmentation.unet.architectures.unet_model import UNet, MultiUNet
from segmentation.unet.architectures.ternaus_net import AlbuNet, UNet16
from segmentation.unet.architectures.senet import DefSENet as SENet


def parse_args():
    parser = argparse.ArgumentParser(description='UNet train')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--target-class', type=int, help='Class which model detects')
    parser.add_argument('--submission', type=str, help='Path to submisssion file')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def update_dataframe(df, fnames, mask_batch, columns=['ImageId_ClassId', 'EncodedPixels']):
    for fname, mask in zip(fnames, mask_batch):
        for cls in range(4):
            subm_fname = '{}_{}'.format(fname, cls+1)
            row_dict = dict(zip(columns, [subm_fname, rle2str(mask2rle(mask[cls]))]))
            df = df.append(row_dict, ignore_index=True)
    return df


def create_submission():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']
    models = {
        'unet': UNet,
        'multiunet': MultiUNet,
        'ternausnet16': UNet16,
        'albunet': AlbuNet,
        'senet': SENet
    }

    model = Model(
        models[config['model']['net']](
            config['model']['input_channels'],
            config['model']['model_classes'],
            is_deconv=config['model']['use_deconv']
        ),
        device
    )

    if config['train']['checkpoint']['use']:
        weight_path = config['train']['checkpoint']['model']
        model.load(weight_path)
        print('Load model: {}'.format(weight_path))
    else:
        raise ValueError('Model weights must be loaded')

    if weight_path is not None:
        model.load(weight_path)

    # Each batch consists of 1 image split into 8 tiles.
    batch_size = 1  # Just available value!
    test_data = DataLoader(
        OneClassSteelTestDatasetGenerator('../../../data/dataset/test_images/', split_on_tiles=True, tiles_number=8),
        batch_size=batch_size,
        num_workers=n_jobs
    )

    # TODO: save masks in files (to visualize)
    # TODO: inference on one image
    # ? TODO: inference on folder (or dataset).

    df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])

    with torch.no_grad():
        for fnames, data in tqdm(test_data):
            data = torch.cat(tuple(data))
            data = data.to(device)
            output = model.predict(data)
            df = update_dataframe(df, fnames, [combine_tiles(output.cpu(), 256, 1600)])
    df.to_csv(args.submission, index=False)


if __name__ == '__main__':
    create_submission()
