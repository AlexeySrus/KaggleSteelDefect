import torch
import tqdm
import os
import re
from generation.gan.utils.losses import l2
from generation.gan.utils.losses import acc as acc_function
from generation.gan.utils.callbacks import add_prefix


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Model:
    def __init__(self,
                 generator_net,
                 discriminator_net,
                 device='cpu',
                 use_spectral_normalization=False,
                 callbacks_list=None):
        self.device = torch.device('cpu' if device == 'cpu' else 'cuda')
        self.generator_model = generator_net.to(self.device)
        self.discriminator_model = discriminator_net.to(self.device)
        self.callbacks = [] if callbacks_list is None else callbacks_list
        self.last_n = 0
        self.last_discriminator_optimiser_state = None
        self.last_generator_optimiser_state = None
        self.use_spectral_normalization = use_spectral_normalization

        self.generator_model.apply(self.weights_init)
        self.discriminator_model.apply(self.weights_init)

    def fit(self,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            epochs=1,
            loss_function=l2,
            validation_loader=None,
            verbose=False,
            init_start_epoch=1,
            acc_f=acc_function,
            generaror_loss_weights=(1.0, 0),
            is_epoch_scheduler=True):
        """
        Model train method
        Args:
            train_loader: DataLoader
            generator_optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            discriminator_optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
            acc_f: function which evaluate accuracy rate,
            generaror_loss_weights: tuple with losses weights between discriminator loss after generation and MSE loss between input image and generated
            is_epoch_scheduler: set True if lr scheduler work by epoch number
        Returns:
        """
        fake_label = 0
        real_label = 1

        generator_scheduler = None
        if type(generator_optimizer) is tuple:
            generator_scheduler = generator_optimizer[1]
            generator_optimizer = generator_optimizer[0]

        discriminator_scheduler = None
        if type(discriminator_optimizer) is tuple:
            discriminator_scheduler = discriminator_optimizer[1]
            discriminator_optimizer = discriminator_optimizer[0]

        for epoch in range(init_start_epoch, epochs + 1):
            self.generator_model.train()
            self.discriminator_model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0
            avg_epoch_acc = 0

            if generator_scheduler is not None and is_epoch_scheduler:
                generator_scheduler.step(epoch)

            if generator_scheduler is not None and is_epoch_scheduler:
                discriminator_scheduler.step(epoch)

            self.last_n = epoch

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, (_x, _y_true) in enumerate(train_loader):
                    self.last_discriminator_optimiser_state = \
                        discriminator_optimizer.state_dict()
                    self.last_generator_optimiser_state = \
                        generator_optimizer.state_dict()

                    x = _x.to(self.device)
                    y_true = _y_true.to(self.device)

                    # step 1
                    discriminator_optimizer.zero_grad()
                    discriminator_output_on_real = self.discriminator_model(
                        y_true).view(-1)

                    real_labels = torch.full(
                        (x.size(0),),
                        real_label,
                        device=self.device
                    )

                    discriminator_loss_on_real = loss_function(
                        discriminator_output_on_real,
                        real_labels
                    )
                    discriminator_loss_on_real.backward()

                    discriminator_average_output_on_real = \
                        discriminator_output_on_real.mean().item()

                    generated_data = self.generator_model(x)

                    discriminator_output_on_fake = self.discriminator_model(
                        generated_data.detach()).view(-1)

                    fake_labels = torch.full(
                        (x.size(0),),
                        fake_label,
                        device=self.device
                    )

                    discriminator_loss_on_fake = loss_function(
                        discriminator_output_on_fake,
                        fake_labels
                    )
                    # if generaror_loss_weights[1] > 0:
                    #     discriminator_loss_on_fake *= generaror_loss_weights[0]
                    #     discriminator_loss_on_fake += \
                    #         torch.nn.functional.mse_loss(
                    #             generated_data, y_true
                    #         ) * generaror_loss_weights[1]

                    discriminator_loss_on_fake.backward()

                    discriminator_average_output_on_fake = \
                        discriminator_output_on_fake.mean().detach()

                    discriminator_total_loss = \
                        discriminator_loss_on_real + discriminator_loss_on_fake

                    discriminator_optimizer.step()

                    acc = (discriminator_average_output_on_fake +
                           discriminator_average_output_on_real) / 2

                    # step 2
                    generator_optimizer.zero_grad()
                    discriminator_output_on_fake_2 = self.discriminator_model(
                        generated_data
                    ).view(-1)
                    discriminator_loss_on_fake_2 = loss_function(
                        discriminator_output_on_fake_2,
                        real_labels
                    )

                    discriminator_loss_on_fake_2.backward()
                    discriminator_average_output_on_fake_2 = \
                        discriminator_output_on_fake_2.mean().item()

                    generator_optimizer.step()

                    acc += discriminator_average_output_on_fake_2
                    acc /= 2


                    pbar.postfix = \
                        'Epoch: {}/{}, loss: {:.8f}, ' \
                        'acc: {:.8f}, d_lr: {:.8f}, g_lr: {:.8f}'.format(
                            epoch,
                            epochs,
                            discriminator_total_loss.item(),
                            acc,
                            get_lr(discriminator_optimizer),
                            get_lr(generator_optimizer)
                        )

                    avg_epoch_loss += discriminator_total_loss.item(
                    ) / y_true.size(0) / batches_count

                    avg_epoch_acc += acc / batches_count

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'loss': discriminator_total_loss.item(
                            ) / y_true.size(0),
                            'loss_2': discriminator_loss_on_fake_2.item(
                            ) / y_true.size(0),
                            'fake_loss': discriminator_loss_on_fake.item(
                            ) / y_true.size(0),
                            'real_loss': discriminator_loss_on_real.item(
                            ) / y_true.size(0),
                            'n': (epoch - 1) * batches_count + i + 1,
                            'img': x,
                            'acc': acc,
                            'mask_true': y_true,
                            'mask_pred': generated_data.detach()
                        })

                    # if self.use_spectral_normalization:
                    #     self.discriminator_model = torch.nn.utils.spectral_norm(
                    #         self.discriminator_model
                    #     )

                    pbar.update(1)

            test_loss = None
            test_acc = None

            if validation_loader is not None:
                test_loss, test_acc = self.evaluate(
                    validation_loader,
                    torch.nn.functional.mse_loss,
                    verbose,
                    acc_f
                )
                self.generator_model.train()

                if generator_scheduler is not None and not is_epoch_scheduler:
                    generator_scheduler.step(test_loss)

                if generator_scheduler is not None and not is_epoch_scheduler:
                    discriminator_scheduler.step(test_loss)

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss,
                    'acc': avg_epoch_acc,
                    'val acc': test_acc,
                    'n': epoch,
                    'generator_optimize_state':
                        generator_optimizer.state_dict(),
                    'discriminator_optimize_state':
                        discriminator_optimizer.state_dict()
                })

    def evaluate(self,
                 test_loader,
                 loss_function=l2,
                 verbose=False,
                 acc_f=acc_function):
        """
        Test model
        Args:
            test_loader: DataLoader
            loss_function: loss function
            verbose: print progress
            acc_f: accuracy function
        Returns:

        """
        self.generator_model.eval()

        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            set_range = tqdm.tqdm(test_loader) if verbose else test_loader
            for _img, _y_true in set_range:
                img = _img.to(self.device)
                y_true = _y_true.to(self.device)

                y_pred = self.generator_model(img)

                loss = loss_function(
                    y_pred,
                    y_true
                )

                acc = acc_f(
                    y_pred,
                    y_true
                )

                test_loss += loss / y_true.size(0) / len(test_loader)
                test_acc += acc / len(test_loader)

        return test_loss, test_acc

    def predict(self,
                full_images_tensor,
                threshold=0.5,
                stride=127,
                verbose=False):
        """
        Model prediction
        Args:
            full_images_tensor: full images tensor

        Returns:
            numpy array with 4 masks

        """
        # full_images_tensor = full_images_tensor.to('cuda')
        #
        # i = 0
        # h, w = full_images_tensor.shape[-2:]
        #
        # predicts = []
        #
        # while i + h <= w:
        #     crop_tensor = full_images_tensor[:, :, :, i:i + h]
        #
        #     predict = self.model(crop_tensor)
        #
        #     predicts.append(
        #         {
        #             'x_offset': i,
        #             'predict': predict.detach() > threshold
        #         }
        #     )
        #
        #     i += stride
        #
        # predicted_tensor = torch.zeros(
        #     size=(
        #         full_images_tensor.shape[0],
        #         4,
        #         *full_images_tensor.shape[-2:]
        #     ),
        #     dtype=predicts[0]['predict'].dtype
        # ).to(self.device)
        #
        # for pred in predicts:
        #     predicted_tensor[
        #         :, :, :, pred['x_offset']:pred['x_offset'] + h
        #     ] += pred['predict']

        # return predicted_tensor
        raise RuntimeError("Don\'t implement model predict method")

    def set_callbacks(self, callbacks_list):
        self.callbacks = callbacks_list

    def save(self, path):
        torch.save(
            self.generator_model.cpu().state_dict(),
            add_prefix(path, 'generator_')
        )
        self.generator_model = self.generator_model.to(self.device)

        torch.save(
            self.discriminator_model.cpu().state_dict(),
            add_prefix(path, 'discriminator_')
        )
        self.discriminator_model = self.discriminator_model.to(self.device)

    def load(self, path, as_dict=False):
        """
        Load generator and discriminator weights by common path
        Args:
            path: path by template {path}/model-{N}.trh and load two models by
            paths {path}/generator_model-{N}.trh and
            {path}/discriminator_model-{N}.trh
            as_dict: load as dictionary

        Returns:
        """
        if not as_dict:
            self.generator_model.load_state_dict(
                torch.load(
                    add_prefix(path, 'generator_'),
                    map_location='cpu'
                )
            )
            self.discriminator_model.load_state_dict(
                torch.load(
                    add_prefix(path, 'discriminator_'),
                    map_location='cpu'
                )
            )
        else:
            self.generator_model.load_state_dict(
                torch.load(
                    add_prefix(path, 'generator_'),
                    map_location='cpu'
                )['model_state']
            )
            self.discriminator_model.load_state_dict(
                torch.load(
                    add_prefix(path, 'discriminator_'),
                    map_location='cpu'
                )['model_state']
            )

        self.generator_model.eval()
        self.generator_model = self.generator_model.to(self.device)

        self.discriminator_model.eval()
        self.discriminator_model = self.discriminator_model.to(self.device)

    def __del__(self):
        for cb in self.callbacks:
            cb.early_stopping(
                {
                    'model': self,
                    'n': self.last_n,
                    'generator_optimize_state':
                        self.last_generator_optimiser_state,
                    'discriminator_optimize_state':
                        self.last_discriminator_optimiser_state
                }
            )

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)


def get_last_epoch_weights_path(checkpoints_dir, log=None):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder
        log: logging, default standard print
    Returns:
        (
            path to current weights file,
            path to current optimiser file,
            current epoch number
        )
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None, None, 0

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('model-\d+.trh', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None, None, 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))

    if log is not None:
        log('LOAD MODEL PATH: {}'.format(
            os.path.join(checkpoints_dir, weights_files_list[0])
        ))

    n = int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), \
           os.path.join(checkpoints_dir, 'optimize_state-{}.trh'.format(n)), n