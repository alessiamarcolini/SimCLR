import sys
import yaml
import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn

from data_preparation import prepare_data


class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


class LogiticRegressionEvaluator(object):

    def __init__(self, n_features, n_classes):
        self.device = self._get_device()
        self.log_regression = LogisticRegression(
            n_features, n_classes).to(self.device)
        self.scaler = preprocessing.StandardScaler()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _normalize_dataset(self, X_train, X_test):
        print("Standard Scaling Normalizer")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            self.log_regression.eval()
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)
                logits = self.log_regression(batch_x)

                predicted = torch.argmax(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            final_acc = 100 * correct / total
            self.log_regression.train()
            return final_acc

    def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test):
        X_train, X_test = self._normalize_dataset(X_train, X_test)

        train = torch.utils.data.TensorDataset(torch.from_numpy(
            X_train), torch.from_numpy(y_train).type(torch.long))
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=396, shuffle=False)

        test = torch.utils.data.TensorDataset(torch.from_numpy(
            X_test), torch.from_numpy(y_test).type(torch.long))
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=512, shuffle=False)
        return train_loader, test_loader

    def train(self, X_train, y_train, X_test, y_test):

        train_loader, test_loader = self.create_data_loaders_from_arrays(
            X_train, y_train, X_test, y_test)

        weight_decay = self._sample_weight_decay()

        optimizer = torch.optim.Adam(
            self.log_regression.parameters(), 3e-4, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_accuracy = 0

        for e in range(200):

            for batch_x, batch_y in train_loader:

                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                logits = self.log_regression(batch_x)

                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

            epoch_acc = self.eval(test_loader)

            if epoch_acc > best_accuracy:
                #print("Saving new model with accuracy {}".format(epoch_acc))
                best_accuracy = epoch_acc
                torch.save(self.log_regression.state_dict(),
                           'log_regression.pth')

        print("--------------")
        print("Done training")
        print("Best accuracy:", best_accuracy)


def main():

    RUN_NAME = 'Apr17_22-45-19_r033c02s01'
    CWD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    DB_EXPORT_DIR = os.path.join(CWD_FOLDER, 'resources', 'db_data_134.csv')
    # We need to merge it to segmentation data in order to get the patient_id we have in deep_features
    SEGMENTATION_DIR = os.path.join(
        CWD_FOLDER, 'resources', 'segmentation_data.csv')
    CHECKPOINTS_FOLDER = os.path.join(
        CWD_FOLDER, 'runs', RUN_NAME, 'checkpoints')

    # Merge DB
    db_data = pd.read_csv(DB_EXPORT_DIR)
    segmentation_data = pd.read_csv(SEGMENTATION_DIR, index_col=False)
    db_merged = pd.merge(db_data, segmentation_data,
                         left_on='study_uid', right_on='study_instance_uid')

    X_train_feature, y_train, X_valid_feature, y_valid = \
        prepare_data(db_merged, CWD_FOLDER, CHECKPOINTS_FOLDER)

    log_regressor_evaluator = LogiticRegressionEvaluator(
        n_features=X_train_feature.shape[1], n_classes=10)

    log_regressor_evaluator.train(X_train_feature.values.astype(
        np.float32), y_train, X_valid_feature.values.astype(np.float32), y_valid)


if __name__ == '__main__':
    main()

