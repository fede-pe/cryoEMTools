""" This module contains the definition of different classes used during the definition and training of the different
models. """
import random
import numpy as np

from tensorflow.keras.utils import Sequence

import utils


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, aliDict, misaliDict, number_batches, batch_size, dim, mode):
        """Initialization"""
        self.aliDict = aliDict
        self.misaliDict = misaliDict

        self.dim = dim
        self.batch_size = batch_size
        self.number_batches = number_batches
        self.number_datasets = len(self.aliDict)

        # Set mode=0 for training and mode=1 for validation
        if mode == 0:
            self.mode = 2
        elif mode == 1:
            self.mode = 3

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.number_batches

    def __getitem__(self, index):
        """Generate one batch of data"""

        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))  # X : (n_samples, *dim)
        y = np.zeros(self.batch_size, dtype=int)

        # With this workaround we ensure to take batch_size elements evenly distributed by the different datasets
        dataset_batch_size = self.batch_size // self.number_datasets
        module_batch_size = self.batch_size % self.number_datasets

        numberOfElementsPerDataset = [dataset_batch_size] * self.number_datasets

        for i in range(module_batch_size):
            numberOfElementsPerDataset[i] += 1

        counter = 0

        for i, key in enumerate(self.aliDict.keys()):
            # Pick numberOfElementsPerDataset/2 elements from ali and misali data
            aliIDsubset = random.sample(self.aliDict[key][self.mode], numberOfElementsPerDataset[i])
            misaliIDsubset = random.sample(self.misaliDict[key][self.mode], numberOfElementsPerDataset[i])

            print("For dataset %s" % key)
            print("shape of ali and mismali subsets")
            print(np.shape(aliIDsubset))
            print(np.shape(misaliIDsubset))

            # Save data
            for j in range(numberOfElementsPerDataset[i]):
                aliIndex = 2 * counter
                misaliIndex = (2 * counter) + 1

                # Store sample
                X[aliIndex, :] = self.aliDict[key][0][aliIDsubset[j], :]
                X[misaliIndex, :] = self.misaliDict[key][0][misaliIDsubset[j], :]

                # Store class
                y[aliIndex] = 1  # Ali
                y[misaliIndex] = 0  # Misali

                counter += 1

        # ---------------------- THE CODE BELLOW IS NOT USABLE AFTER THE LAST MODIFICATION

        # Generate phantom data (Ali as it is, Misali = Ali * -1)
        # for i in range(self.batch_size // 2):
        #     aliIndex = 2 * i
        #     misaliIndex = (2 * i) + 1
        #
        #     # Store sample
        #     X[aliIndex, :] = self.aliData[aliIDsubset[i], :]
        #     X[misaliIndex, :] = self.aliData[aliIDsubset[i], :] * -1
        #
        #     # Store class
        #     y[aliIndex] = 1  # Ali
        #     y[misaliIndex] = 0  # Misali

        # # Generate phantom data (Ali as it is, Misali = rot90z of ali)
        # for i in range(self.batch_size // 2):
        #     from math import cos, sin
        #     import xmippLib as xmipp
        #
        #     Z_ROTATION_90 = np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0],
        #                                 [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0],
        #                                 [0, 0, 1]])
        #
        #     aliIndex = 2 * i
        #     misaliIndex = (2 * i) + 1
        #
        #     subtomo = self.aliData[aliIDsubset[i], :]
        #     inputSubtomo = xmipp.Image()
        #     outputSubtomo = np.zeros(self.dim)
        #
        #     for slice in range(self.dim[0]):
        #         inputSubtomo.setData(subtomo[slice])
        #
        #         outputSubtomoImage = inputSubtomo.applyWarpAffine(list(Z_ROTATION_90.flatten()),
        #                                                           self.dim,
        #                                                           True)
        #
        #         outputSubtomo[slice] = outputSubtomoImage.getData()
        #
        #     # Store sample
        #     X[aliIndex, :] = self.aliData[aliIDsubset[i], :]
        #     X[misaliIndex, :] = outputSubtomo
        #
        #     # Store class
        #     y[aliIndex] = 1  # Ali
        #     y[misaliIndex] = 0  # Misali

        # Testing the numpy array generation
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
