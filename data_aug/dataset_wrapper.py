import copy

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from data_aug.gaussian_blur import GaussianBlur

from .dataset_triplets import CovidDatasetPatient
from .split import train_test_indexes_patient_wise

np.random.seed(0)


class DataSetWrapper(object):
    def __init__(self, dataset_dir, batch_size, num_workers, valid_size, input_shape, s):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self, original=True, augment=True):
        if original:
            data_augment = self._get_simclr_pipeline_transform()
        else:
            data_augment = self._get_simclr_pipeline_transform_covid()

        if original:
            if augment:
                train_dataset = datasets.STL10(
                    './data',
                    split='train+unlabeled',
                    download=True,
                    transform=SimCLRDataTransform(data_augment),
                )
            else:
                train_dataset = datasets.STL10(
                    './data', split='train+unlabeled', download=True,
                )
        else:
            if augment:
                train_dataset = CovidDatasetPatient(
                    self.dataset_dir, transform=data_augment
                )
            else:
                train_dataset = CovidDatasetPatient(
                    self.dataset_dir
                )

        train_loader, valid_loader = self.get_train_validation_data_loaders(
            train_dataset, original=original
        )
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s
        )
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.input_shape[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def _get_simclr_pipeline_transform_covid(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(
            brightness=0.3 * self.s, contrast=0.3 * self.s
        )
        data_transforms = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(-10, 10),
                    translate=(0.1, 0.1),
                    scale=(0.80, 1.20), 
                    shear=(-10, 10)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset, original=True):

        if original:
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]

            # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
                drop_last=True,
                shuffle=False,
            )

            valid_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=valid_sampler,
                num_workers=self.num_workers,
                drop_last=True,
            )

        else:
            train_indices, valid_indices = train_test_indexes_patient_wise(
                train_dataset, self.valid_size, 1234
            )

            # valid_dataset = CovidDatasetPatient(
            #    '/thunderdisk/covid/dataset_triplets', transform=data_augment
            # )
            valid_dataset = copy.deepcopy(train_dataset)
            train_dataset.indices = np.array(train_indices)
            valid_dataset.indices = np.array(valid_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )
            print(len(train_dataset), len(valid_dataset))

        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
