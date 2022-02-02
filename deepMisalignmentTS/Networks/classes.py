""" This module contains the definition of different classes used during the definition and training of the different
models. """
import random

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, aliData, misaliData, aliIDs, misaliIDs, number_batches, batch_size, dim):
        """Initialization"""
        self.aliData = aliData
        self.misaliData = misaliData
        self.aliIDs = aliIDs
        self.misaliIDs = misaliIDs

        self.dim = dim
        self.batch_size = batch_size
        self.number_batches = number_batches

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.number_batches

    def __getitem__(self, index):
        """Generate one batch of data"""

        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Pick batch_size/2 elements from ali and misali data vectors (2 * batch_size/2)
        aliIDsubset = random.sample(self.aliIDs, self.batch_size / 2)
        misaliIDsubset = random.sample(self.misaliIDs, self.batch_size / 2)

        print("SHAPE OF ALI AND MISMALI SUBSETS")
        print(np.shape(aliIDsubset))
        print(np.shape(misaliIDsubset))

        # Generate data
        for i, in range(len(aliIDsubset)):
            # Store sample
            X[2 * i, ] = self.aliData[aliIDsubset[i], :]
            X[(2 * i) + 1, ] = self.misaliData[misaliIDsubset[i], :]

            # Store class
            y[2 * i] = 1  # Ali
            y[(2 * i) + 1, ] = 0  # Misali

        return X, y
