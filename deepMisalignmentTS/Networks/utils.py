""" Module containing functions to parse and process datasets and results. """

import numpy as np
from math import cos, sin

import xmippLib as xmipp

Z_ROTATION_180 = np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0],
                             [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0],
                             [0, 0, 1]])

Y_ROTATION_180 = np.asarray([[cos(np.deg2rad(180)), 0, sin(np.deg2rad(180))],
                             [0, 1, 0],
                             [-sin(np.deg2rad(180)), 0, cos(np.deg2rad(180))]])

Z_Y_ROTATION_180 = np.matmul(Z_ROTATION_180, Y_ROTATION_180)

_MATRICES = [Z_ROTATION_180, Y_ROTATION_180, Z_Y_ROTATION_180]

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


def dataAugmentationSubtomo(subtomo, aligment, foldAugmentation, shape):
    """ This methods takes a subtomo used as a reference and returns a rotated version of this for data augmentation.
    Given a subtomo there is only 3 possible transformation (the combination of 180º rotations in Y and Z axis) in order
    to match the missing wedge between the input and the output subtomo.
    :param subtomo: input reference subtomo.
    :param aligment: alignment/misalignment toggle of the subtomo
    :param foldAugmentation: number of subtomos generated per each input reference.
    :param shape: output shape of the subtomos.
    """

    # There is a maximum of 3 possible transformations
    if foldAugmentation > 2:
        foldAugmentation = 2

    # Subtomo data augmentation
    outputSubtomos = []

    inputSubtomo = xmipp.Image()

    outputSubtomo = np.zeros(shape)

    for i in range(foldAugmentation):
        for slice in range(shape[2]):
            inputSubtomo.setData(subtomo[slice])

            outputSubtomoImage = inputSubtomo.applyWarpAffine(list(_MATRICES[i].flatten()),
                                                              shape,
                                                              True)

            outputSubtomo[slice] = outputSubtomoImage.getData()

        outputSubtomos.append(outputSubtomo)

    # Testing the transformation
    outputSubtomoImage.setData(outputSubtomo)

    inputSubtomo.setData(subtomo)

    inputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testIn.mrc"
    outputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testOut.mrc"

    inputSubtomo.write(inputFilePath)
    outputSubtomoImage.write(outputFilePath)

    # Alignment data augmentation
    outputMisalignment = []

    for i in range(foldAugmentation):
        outputMisalignment.append(aligment)

    return outputSubtomos, outputMisalignment
