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
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros(self.batch_size, dtype=int)

        # Pick batch_size/2 elements from ali and misali data vectors (2 * batch_size/2)
        aliIDsubset = random.sample(self.aliIDs, self.batch_size // 2)
        misaliIDsubset = random.sample(self.misaliIDs, self.batch_size // 2)

        # print("SHAPE OF ALI AND MISMALI SUBSETS")
        # print(np.shape(aliIDsubset))
        # print(np.shape(misaliIDsubset))

        # print("SHAPE OF X AND y")
        # print(np.shape(X))
        # print(np.shape(y))
        #
        # print(len(self.aliIDs))
        # print(len(self.misaliIDs))

        # Generate data
        for i in range(self.batch_size//2):
            aliIndex = 2 * i
            misaliIndex = (2 * i) + 1

            # Store sample
            X[aliIndex, :] = self.aliData[aliIDsubset[i], :]
            X[misaliIndex, :] = self.misaliData[misaliIDsubset[i], :]

            # Store class
            y[aliIndex] = 1  # Ali
            y[misaliIndex] = 0  # Misali

        # Testing the numpy array generation
        #
        # import xmippLib as xmipp
        # inputSubtomoArray1 = xmipp.Image()
        # inputSubtomoArray2 = xmipp.Image()
        # inputSubtomo1 = xmipp.Image()
        # inputSubtomo2 = xmipp.Image()
        #
        # for i in range(self.batch_size//2):
        #     aliIndex = 2 * i
        #     misaliIndex = (2 * i) + 1
        #
        #     X_tmp = X[aliIndex, :]
        #     inputSubtomoArray1.setData(X_tmp)
        #     inputSubtomoArray1.write(str(aliIndex)+"_nparray_subtomo.mrc")
        #
        #     X_tmp = X[misaliIndex, :]
        #     inputSubtomoArray2.setData(X_tmp)
        #     inputSubtomoArray2.write(str(misaliIndex)+"_nparray_subtomo.mrc")
        #
        #     inputSubtomo1.setData(self.aliData[aliIDsubset[i], :])
        #     inputSubtomo1.write(str(aliIndex)+"_subtomo.mrc")
        #
        #     inputSubtomo2.setData(self.misaliData[misaliIDsubset[i], :])
        #     inputSubtomo2.write(str(misaliIndex)+"_subtomo.mrc")

        return X, y
