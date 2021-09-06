""" Module containing functions to parse and process datasets and results. """

import numpy as np
from math import cos, sin

import xmippLib as xmipp

Z_ROTATION_180 = np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0, 0],
                             [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

Y_ROTATION_180 = np.asarray([[cos(np.deg2rad(180)), 0, sin(np.deg2rad(180)), 0],
                             [0, 1, 0, 0],
                             [-sin(np.deg2rad(180)), 0, cos(np.deg2rad(180)), 0],
                             [0, 0, 0, 1]])


def normalizeInputDataStream(inputSubtomoStream):
    """ Method to normalize the input subtomo data stream to """
    std = inputSubtomoStream.std()
    mean = inputSubtomoStream.mean()

    normalizedInputDataStream = (inputSubtomoStream - mean) / std

    return normalizedInputDataStream


def produceClassesDistributionInfo(misalignmentInfoVector, verbose=True):
    """ This method output information of the classes distributions from the dataset between aligned and misaligned
    subtomos. """

    totalSubtomos = len(misalignmentInfoVector)
    numberOfMisalignedSubtomos = 0
    numberOfAlignedSubtomos = 0

    for subtomo in misalignmentInfoVector:
        if subtomo == 0:
            numberOfMisalignedSubtomos += 1
        elif subtomo == 1:
            numberOfAlignedSubtomos += 1

    if verbose:
        print("\nClasses distribution:\n"
              "Aligned: %d (%.3f%%)\n"
              "Misaligned: %d (%.3f%%)\n\n"
              % (numberOfAlignedSubtomos, (numberOfAlignedSubtomos / totalSubtomos) * 100,
                 numberOfMisalignedSubtomos, (numberOfMisalignedSubtomos / totalSubtomos) * 100))

    return numberOfAlignedSubtomos / totalSubtomos


def dataAugmentationSubtomo(subtomo, foldAugmentation, shape):
    """ This methods takes a subtomo used as a reference and returns a rotated version of this for data augmentation.
    Given a subtomo there is only 3 possible transformation (the combination of 180º rotations in Y and Z axis) in order
    to match the missing wedge between the input and the output subtomo.
    :param subtomo: input reference subtomo.
    :param foldAugmentation: number of subtomos generated per each input reference.
    :param shape: otuput shape of the subtomos.
    """

    matrices = [Z_ROTATION_180, Y_ROTATION_180, np.matmul(Z_ROTATION_180, Y_ROTATION_180)]
    outputSubtomos = []

    rotM = np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0],
                       [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0],
                       [0, 0, 1]])

    np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0, 0],
                [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    subtomoTest = subtomo[16, :, :]

    imagTest = xmipp.Image()
    imagTest.setData(subtomoTest)

    imagTest.write("/home/fede/cryoEMTools/deepMisalignmentTS/Networks/imageTestIn.mrc")

    resultImageTest = imagTest.applyWarpAffine(list(rotM.flatten()), subtomoTest.shape, False, 0)

    resultImageTest.write("/home/fede/cryoEMTools/deepMisalignmentTS/Networks/imageTestOut.mrc")

    print(subtomo.shape)
    print(subtomo)

    for i in range(foldAugmentation):
        M = matrices[i]

        print(M)
        print(list(M.flatten()))

        imag = xmipp.Image()
        imag.setData(subtomo)
        imag.write("/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testIn.mrc")

        resultSubtomo = imag.applyWarpAffine(list(M.flatten()), subtomo.shape, False, 0)

        resultSubtomo.write("/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testOut.mrc")

        outputSubtomos.append(resultSubtomo.getData())

    return outputSubtomos
