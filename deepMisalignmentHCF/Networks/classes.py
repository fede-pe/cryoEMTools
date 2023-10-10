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
        self.number_datasets_ali = len(self.aliDict)
        self.number_datasets_misali = len(self.misaliDict)

        # Set mode=0 for training and mode=1 for validation
        if mode == 0:
            print("Data generator training class created:")
            self.mode = 2
        elif mode == 1:
            print("Data generator validation class created:")
            self.mode = 3

        print("self.dim " + str(self.dim))
        print("self.batch_size " + str(self.batch_size))
        print("self.number_batches " + str(self.number_batches))
        print("self.number_datasets_ali " + str(self.number_datasets_ali))
        print("self.number_datasets_misali " + str(self.number_datasets_misali))
        print("self.mode " + str(self.mode))

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
        # ensuring that half of them are aligned and the other half misaligned
        dataset_batch_size_half = self.batch_size // 2
        dataset_batch_size_ali = dataset_batch_size_half // self.number_datasets_ali
        dataset_batch_size_misali = dataset_batch_size_half // self.number_datasets_misali

        module_batch_size_ali = dataset_batch_size_half % self.number_datasets_ali
        module_batch_size_misali = dataset_batch_size_half % self.number_datasets_misali

        numberOfElementsPerDataset_ali = [dataset_batch_size_ali] * self.number_datasets_ali
        for i in range(module_batch_size_ali//2):
            numberOfElementsPerDataset_ali[i] += 2

        numberOfElementsPerDataset_misali = [dataset_batch_size_misali] * self.number_datasets_misali
        for i in range(module_batch_size_misali//2):
            numberOfElementsPerDataset_misali[i] += 2

        print("Number of elements selected for each vector:")
        print("numberOfElementsPerDataset_ali " + str(numberOfElementsPerDataset_ali))
        print("numberOfElementsPerDataset_misali" + str(numberOfElementsPerDataset_misali))

        # Fill with ali data
        counter = 0

        for i, key in enumerate(self.aliDict.keys()):
            # Pick numberOfElementsPerDataset_ali from ali data
            print("-- Data generation for dataset %s " % key)
            print("Number of elements to be taken from this vector %d " % numberOfElementsPerDataset_ali[i])
            print("Size of the vector %d " % len(self.aliDict[key][self.mode]))

            aliIDsubset = random.sample(self.aliDict[key][self.mode], numberOfElementsPerDataset_ali[i])

            # Save data
            for j in range(numberOfElementsPerDataset_ali[i]):
                aliIndex = 2 * counter

                # Store sample
                X[aliIndex, :] = self.aliDict[key][0][aliIDsubset[j], :]

                # Store class
                y[aliIndex] = 1  # Ali

                counter += 1

        # Fill with misali data
        counter = 0

        for i, key in enumerate(self.misaliDict.keys()):
            # Pick numberOfElementsPerDataset/2 elements from misali data
            print("-- Data generation for dataset %s " % key)
            print("Number of elements to be taken from this vector %d " % numberOfElementsPerDataset_misali[i])
            print("Size of the vector %d " % len(self.misaliDict[key][self.mode]))

            misaliIDsubset = random.sample(self.misaliDict[key][self.mode], numberOfElementsPerDataset_misali[i])

            # Save data
            for j in range(numberOfElementsPerDataset_misali[i]):
                misaliIndex = (2 * counter) + 1

                # Store sample
                X[misaliIndex, :] = self.misaliDict[key][0][misaliIDsubset[j], :]

                # Store class
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
