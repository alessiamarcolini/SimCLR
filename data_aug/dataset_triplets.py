import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CovidDatasetPatient(Dataset):
    def __init__(self, dataset_dir, transform=None, size=(256, 256)):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.size = size

        filenames = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npy')])
        self._filenames = np.array(filenames)

        patients = [f.split('_')[0] for f in filenames]
        self._patients = np.array(patients)

        self.indices = np.arange(len(self._filenames))
        # print(len(filenames))

    def __getitem__(self, idx):
        filename = self._filenames[self.indices[idx]]
        patient = self._patients[self.indices[idx]]
        image = np.load(os.path.join(self.dataset_dir, filename))
        # print(image.shape)
        image = np.moveaxis(image, 0, 2)
        # print(image.shape)
        image = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')

        if self.transform:
            xi = self.transform(image)
            xj = self.transform(image)
            return (xi, xj), {'patient': patient, 'filename': filename}

        else:
            image = np.array(image)
            image = np.moveaxis(image, 2, 0)
            out = torch.Tensor(image)
            return (
                (out, out),
                {'patient': patient, 'filename': filename},
            )  # if no augmentation, return the same image twice ?

        # return {'image': out, 'filename': filename, 'patient': patient}

    @property
    def filenames(self):
        return self._filenames[self.indices]

    @property
    def patients(self):
        return self._patients[self.indices]

    def __len__(self):
        return len(self.indices)

    def __shuffle__(self):
        idx_permut = np.random.permutation(self.__len__())
        self._filenames = self._filenames[idx_permut]
        self._patients = self._patients[idx_permut]
        self.indices = self.indices[idx_permut]
