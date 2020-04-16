import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from loss.nt_xent import NTXentLoss
from models.resnet_simclr import ResNetSimCLR

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False


torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            './config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml')
        )


class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config['batch_size'], **config['loss']
        )

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders(
            self.config['original']
        )

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay'])
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        self.model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(self.model_checkpoints_folder)

        n_iter_global = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            n_iter = 0

            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter_global)

                if n_iter_global % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar(
                        'train_loss', loss, global_step=n_iter_global
                    )

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter_global += 1
                n_iter += 1
                sys.stdout.write(
                    "\r Epoch {} of {}  [{:.2f}%] - loss TR {:.4f}".format(
                        epoch_counter + 1,
                        self.config['epochs'],
                        100 * n_iter / len(train_loader),
                        loss.item(),
                    )
                )

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.model_checkpoints_folder, 'model.pth'),
                    )

                self.writer.add_scalar(
                    'validation_loss', valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar(
                'cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter
            )

        self.model = model

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                './runs', self.config['fine_tune_from'], 'checkpoints'
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            for counter, ((xis, xjs), _) in enumerate(valid_loader):
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()

            valid_loss = valid_loss / (counter + 1)
        model.train()
        return valid_loss

    def extract_features(self, save=True):

        model = self.model

        train_loader, valid_loader = self.dataset.get_data_loaders(
            self.config['original'], augment=False
        )
        print(len(train_loader), len(valid_loader))

        X_train_feature = []
        filenames_train = []
        patients_train = []

        X_valid_feature = []
        filenames_valid = []
        patients_valid = []

        with torch.no_grad():
            model.eval()

            for counter, ((image, _), info) in enumerate(train_loader):
                image = image.to(self.device)
                filename_batch = info['filename']
                patient_batch = info['patient']

                filenames_train.append(filename_batch)
                patients_train.append(patient_batch)

                features = model.extract_features(image)
                X_train_feature.extend(features.cpu().detach().numpy())

            X_train_feature = np.squeeze(np.array(X_train_feature))
            filenames_train = np.concatenate(filenames_train)
            patients_train = np.concatenate(patients_train)

            print('Features train shape: ', X_train_feature.shape)

        with torch.no_grad():
            model.eval()

            for counter, ((image, _), info) in enumerate(valid_loader):
                image = image.to(self.device)
                filename_batch = info['filename']
                patient_batch = info['patient']

                filenames_valid.append(filename_batch)
                patients_valid.append(patient_batch)

                features = model.extract_features(image)
                X_valid_feature.extend(features.cpu().detach().numpy())

            X_valid_feature = np.squeeze(np.array(X_valid_feature))
            filenames_valid = np.concatenate(filenames_valid)
            patients_valid = np.concatenate(patients_valid)

            print('Features valid shape: ', X_valid_feature.shape)
            print('filenames valid shape: ', filenames_valid.shape)

        if save:

            os.makedirs(
                os.path.join(self.model_checkpoints_folder, 'deep_features'),
                exist_ok=True,
            )

            train_features_df = pd.DataFrame(X_train_feature)
            train_features_df['patient'] = patients_train
            train_features_df['filename'] = filenames_train

            valid_features_df = pd.DataFrame(X_valid_feature)
            valid_features_df['patient'] = patients_valid
            valid_features_df['filename'] = filenames_valid

            train_features_df.to_csv(
                os.path.join(
                    self.model_checkpoints_folder, 'deep_features', 'features_train.csv'
                ),
                index=False,
            )
            valid_features_df.to_csv(
                os.path.join(
                    self.model_checkpoints_folder, 'deep_features', 'features_valid.csv'
                ),
                index=False,
            )
