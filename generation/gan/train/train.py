import torch
import argparse
import os
import yaml
from shutil import copyfile
import torch.nn.functional as F
from generation.gan.utils.optimizers import Nadam
from generation.gan.model.model import Model, get_last_epoch_weights_path
from generation.gan.utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                      SaveOptimizerPerEpoch,
                                                  VisImage, ModelLogging,
                                                  TensorboardPlotCallback)
from generation.gan.dataset.dataset_generator import\
    SteelDatasetGenerator
from generation.gan.utils.losses import l2, iou_acc
from torch.utils.data import DataLoader
from generation.gan.achitectures.gan import Generator, Discriminator
from generation.gan.achitectures.autorncoder import VideoImprovingNet


def parse_args():
    parser = argparse.ArgumentParser(description='UNet train')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']
    epochs = config['train']['epochs']

    losses = {
        'l2': l2,
        'l1': F.l1_loss,
        'mse': F.mse_loss,
        'bn': torch.nn.CrossEntropyLoss(),
        'bce': torch.nn.BCEWithLogitsLoss()
    }

    optimizers = {
        'adam': torch.optim.Adam,
        'nadam': Nadam,
        'sgd': torch.optim.SGD,
    }

    if not os.path.isdir(config['train']['save']['model']):
        os.makedirs(config['train']['save']['model'])

    copyfile(
        args.config,
        os.path.join(
            config['train']['save']['model'],
            os.path.basename(args.config)
        )
    )

    generator_model = VideoImprovingNet(
        2,
        True,
        1,
        32
    )
    # generator_model = Generator(
    #     config['model']['input_channels'],
    #     config['model']['model_classes'],
    #     config['model']['discriminator_features_map_size']
    # )

    discriminator_model = Discriminator(
        config['model']['model_classes'],
        config['model']['generator_features_map_size']
    )

    model = Model(
        generator_model,
        discriminator_model,
        device,
        use_spectral_normalization=config['model']['use_spectral_normalization']
    )

    callbacks = []

    callbacks.append(SaveModelPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    callbacks.append(SaveOptimizerPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    if config['logging']['use_logger']:
        callbacks.append(
            ModelLogging(
                path=config['logging']['log_path'],
                save_step=config['logging']['log_step'],
                columns=['n', 'loss'],
                continue_train=config['train']['load']
            )
        )

    if config['visualization']['use_visdom']:
        plots = VisPlot(
            'UNet',
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        plots.register_scatterplot('train loss per_batch',
                                   'Batch number',
                                   'Loss',
                                   [
                                       '{} loss of step 1'.format(
                                           config['train']['loss'].upper()),
                                       '{} loss of step 2'.format(
                                           config['train']['loss'].upper())
                                   ])

        plots.register_scatterplot('train validation loss per_epoch',
                                   'Epoch number',
                                   'Loss',
                                   [
                                       'train loss', 'val loss'
                                   ])

        plots.register_scatterplot('train validation acc per_epoch',
                                   'Epoch number',
                                   'mIOU',
                                   [
                                       'train acc', 'val acc'
                                   ])

        callbacks.append(plots)

        callbacks.append(
            VisImage(
                'Image visualisation',
                config['visualization']['visdom_server'],
                config['visualization']['visdom_port'],
                config['visualization']['image']['every'],
                scale=config['visualization']['image']['scale']
            )
        )

    if config['visualization']['use_tensorboard']:
        callbacks.append(
            TensorboardPlotCallback(
                log_dir=config['visualization']['tensorboard_logdir'],
                loss_name=config['train']['loss'],
                save_params=config['visualization']['save_historgam']
            )
        )

    model.set_callbacks(callbacks)

    start_epoch = 0

    if config['train']['optimizer'] != 'sgd':
        generator_optimizer = optimizers[config['train']['optimizer']](
            model.generator_model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        discriminator_optimizer = optimizers[config['train']['optimizer']](
            model.discriminator_model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        generator_optimizer = optimizers[config['train']['optimizer']](
            model.generator_model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        discriminator_optimizer = torch.optim.SGD(
            model.discriminator_model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'],
            momentum=0.9,
            nesterov=True

        )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     verbose=True
    # )

    weight_path = None
    optim_path = None

    if config['train']['checkpoint']['use']:
        weight_path = config['train']['checkpoint']['model']
        optim_path = config['train']['checkpoint']['optimizer']
        print('Start from checkpoint: {}'.format(weight_path))
    else:
        if config['train']['load']:
            weight_path, optim_path, start_epoch = get_last_epoch_weights_path(
                os.path.join(
                    os.path.dirname(__file__),
                    config['train']['save']['model']
                ),
                print
            )

    if weight_path is not None:
        model.load(weight_path)
        generator_optimizer.load_state_dict(
            torch.load(optim_path, map_location='cpu')
        )
        discriminator_optimizer.load_state_dict(
            torch.load(optim_path, map_location='cpu')
        )

    train_data = DataLoader(
        SteelDatasetGenerator(
            dataset_path=config['dataset']['train_images_path'],
            table_path=config['dataset']['train_table_path'],
            validation=False,
            validation_part=config['dataset']['validation_part']
        ),
        batch_size=batch_size,
        num_workers=n_jobs,
        shuffle=True,
        drop_last=True
    )

    validation_data = DataLoader(
        SteelDatasetGenerator(
            dataset_path=config['dataset']['train_images_path'],
            table_path=config['dataset']['train_table_path'],
            validation=True,
            validation_part=config['dataset']['validation_part']
        ),
        batch_size=batch_size,
        num_workers=n_jobs
    )

    model.fit(
        train_data,
        generator_optimizer,
        discriminator_optimizer,
        epochs,
        losses[config['train']['loss']],
        init_start_epoch=start_epoch + 1,
        validation_loader=validation_data,
        acc_f=iou_acc,
        is_epoch_scheduler=False
    )


if __name__ == '__main__':
    main()
