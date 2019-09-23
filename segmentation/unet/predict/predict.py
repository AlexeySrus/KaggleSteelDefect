import torch
import argparse
import yaml
import pandas as pd
from segmentation.unet.utils.image_utils import rle2str, mask2rle
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
            subm_fname = '{}_{}'.format(fname, cls)
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

    generated_train_dataset = OneClassSteelDatasetGenerator(
        dataset_path=config['dataset']['train_images_path'],
        table_path=config['dataset']['train_table_path'],
        class_index=config['dataset']['select_class'],
        part_without_masks_relatively_with_masks=
        config['dataset']['part_without_masks_relatively_with_masks'],
        validation=False,
        validation_part=config['dataset']['validation_part'],
        augmentation=config['train']['augmentation'],
        preused_table=None
    )

    train_data = DataLoader(
        generated_train_dataset,
        batch_size=batch_size,
        num_workers=n_jobs,
        shuffle=True,
        drop_last=True
    )

    validation_data = DataLoader(
        OneClassSteelDatasetGenerator(
            dataset_path=config['dataset']['train_images_path'],
            table_path=config['dataset']['train_table_path'],
            class_index=config['dataset']['select_class'],
            part_without_masks_relatively_with_masks=
            config['dataset']['part_without_masks_relatively_with_masks'],
            validation=True,
            validation_part=config['dataset']['validation_part'],
            preused_table=generated_train_dataset.fixed_table_data
        ),
        batch_size=batch_size,
        num_workers=n_jobs
    )

    test_data = DataLoader(
        OneClassSteelTestDatasetGenerator('../../../data/dataset/test_images/'),
        batch_size=batch_size,
        num_workers=n_jobs
    )

    # TODO: Split image on 256x256 tiles and form back
    # TODO: save masks in files (to visualize)
    # TODO: inference on one image
    # ? TODO: inference on folder (or dataset).

    # PROBLEM: result file has size of 2,5 GB


    df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])

    with torch.no_grad():
        for fnames, data in test_data:
            data = data.to(device)
            output = model.predict(data)
            df = update_dataframe(df, fnames, output.cpu())
    df.to_csv(args.submission)

    # with torch.no_grad():
    #     for data, target in validation_data:
    #         data = data.to(device)
    #         print(data.shape)
    #         output = model.predict(data)
    #         print(output.shape)


if __name__ == '__main__':
    create_submission()
