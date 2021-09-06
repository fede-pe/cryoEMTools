
""" Module containing functions to parse and process datasets and results. """

import numpy as np
from math import cos, sin

import xmippLib as xmipp


Z_ROTATION_180 = np.asarray([cos(180), -sin(180), 0, 0],
                            [sin(180), cos(180), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1])

Y_ROTATION_180 = np.asarray([cos(180), 0, sin(180), 0],
                            [0, 1, 0, 0],
                            [-sin(180), 0, cos(180), 0],
                            [0, 0, 0, 1])


def normalizeInputDataStream(inputSubtomoStream):
    """ Method to normalize the input subtomo data stream to """
    std = inputSubtomoStream.std()
    mean = inputSubtomoStream.mean()

    normalizedInputDataStream = (inputSubtomoStream - mean) / std

    return normalizedInputDataStream


def produceClassesDistributionInfo(misalignmentInfoVector):
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

    print("\nClasses distribution:\n"
          "Aligned: %d (%.3f%%)\n"
          "Misaligned: %d (%.3f%%)\n\n"
          % (numberOfAlignedSubtomos, (numberOfAlignedSubtomos / totalSubtomos) * 100,
             numberOfMisalignedSubtomos, (numberOfMisalignedSubtomos / totalSubtomos) * 100))


def rotateSubtomo(subtomo, foldAugmentation, shape):
    """ This methods takes a subtomo used as a reference and returns a rotated version of this for data augmentation.
    Given a subtomo there is only 3 possible transformation (the combination of 180º rotations in Y and Z axis) in order
    to match the missing wedge between the input and the output subtomo.
    :param subtomo: input reference subtomo.
    :param augmentation: number of subtomos generated per each input reference. """

    matrices = [Z_ROTATION_180, Y_ROTATION_180, np.matmul(Z_ROTATION_180, Y_ROTATION_180)]
    outputSubtomos = []

    for i in range(foldAugmentation):
        M = matrices[i]

        imag = xmipp.Image()
        imag.setData(subtomo)

        imag = imag.applyWarpAffine(list(M.flatten()), shape, True)

        outputSubtomos.append(imag.getData())

    return outputSubtomos
