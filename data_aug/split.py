import copy

import numpy as np
from sklearn.model_selection import train_test_split


def train_test_indexes_patient_wise(dataset, test_size=0.2, seed=1234):

    patients = dataset.patients
    unique_patients = np.unique(patients)

    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=seed
    )

    train_indexes = []
    test_indexes = []

    for train_patient in train_patients:
        idxs = np.where(train_patient == np.array(patients))[0].tolist()
        train_indexes.extend(idxs)

    for test_patient in test_patients:
        idxs = np.where(test_patient == np.array(patients))[0].tolist()
        test_indexes.extend(idxs)

    return train_indexes, test_indexes
