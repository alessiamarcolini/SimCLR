import sys
import os
import yaml

import numpy as np
import pandas as pd


def prepare_data(db_merged, cwd_folder, checkpoints_folder):
    
    # Read CSV
    features_train = pd.read_csv(os.path.join(
        checkpoints_folder, 'deep_features', 'features_train.csv'))
    features_valid = pd.read_csv(os.path.join(
        checkpoints_folder, 'deep_features', 'features_valid.csv'))

    # Get deep features
    X_train_feature = features_train.iloc[:, :-2]
    X_valid_feature = features_valid.iloc[:, :-2]

    # Get patients name
    # These are pd.Series -> we should use those same indexes
    patients_train = features_train.iloc[:, -2]
    patients_valid = features_valid.iloc[:, -2]

    # Get y_labels for training and validation set
    y_train = []
    for p in patients_train:
        y_train.append(
            db_merged[db_merged['patient_id'] == p]['consolidations'].values[0]
        )
    y_train = np.array(y_train)

    y_valid = []
    for p in patients_valid:
        y_valid.append(
            db_merged[db_merged['patient_id'] == p]['consolidations'].values[0]
        )
    y_valid = np.array(y_valid)

    return X_train_feature, y_train, X_valid_feature, y_valid