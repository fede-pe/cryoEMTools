""" Module containing functions to parse and process datasets and results. """

import numpy as np
from math import cos, sin
import random

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
    """ Method to normalize the input subtomo data stream element by element """
    for n, _ in enumerate(inputSubtomoStream):
        inputSubtomoStream[n] = (inputSubtomoStream[n] - inputSubtomoStream[n].mean()) / inputSubtomoStream[n].std()

    return inputSubtomoStream


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


def dataAugmentationSubtomo(subtomo, alignment, shape):
    """ This methods takes a subtomo used as a reference and returns a rotated version of this for data augmentation.
    Given a subtomo there is only 3 possible transformation (the combination of 180º rotations in Y and Z axis) in order
    to match the missing wedge between the input and the output subtomo.
    :param subtomo: input reference subtomo.
    :param alignment: alignment/misalignment toggle of the subtomo
    :param shape: output shape of the subtomos.
    """

    # There is a maximum of 3 possible transformations
    foldAugmentation = 3

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
        # if i == 2:
        #     outputSubtomoImage.setData(outputSubtomo)
        #
        #     inputSubtomo.setData(subtomo)
        #
        #     inputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testIn.mrc"
        #     outputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testOut.mrc"
        #
        #     inputSubtomo.write(inputFilePath)
        #     outputSubtomoImage.write(outputFilePath)

    # Alignment data augmentation
    outputMisalignment = []

    for i in range(foldAugmentation):
        outputMisalignment.append(alignment)

    return outputSubtomos, outputMisalignment


def dataAugmentationSubtomoDynamic(subtomo, shape):
    """ This method takes a subtomo used as a reference and returns a rotated version of this for data augmentation
    in dynamic training mode.
    Given a subtomo there is only 3 possible transformation (the combination of 180º rotations in Y and Z axis) in order
    to match the missing wedge between the input and the output subtomo.
    :param subtomo: input reference subtomo.
    :param shape: output shape of the subtomos.
    """

    # There is a maximum of 3 possible transformations
    foldAugmentation = 3

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
        # if i == 2:
        #     outputSubtomoImage.setData(outputSubtomo)
        #
        #     inputSubtomo.setData(subtomo)
        #
        #     inputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testIn.mrc"
        #     outputFilePath = "/home/fede/cryoEMTools/deepMisalignmentTS/Networks/testOut.mrc"
        #
        #     inputSubtomo.write(inputFilePath)
        #     outputSubtomoImage.write(outputFilePath)

    return outputSubtomos


def generateTrainingValidationVectors(size, validationRatio):
    """ Generates two vector of ID's indicating the indices of the subtomos leading to training and validation
    respectively. The partition of length=size*validation ratio is always returned first."""

    # Vector containing the indices of a vector of given size in random order
    randomIndexes = list(range(0, size))
    random.shuffle(randomIndexes)

    limitRatio = int(size * validationRatio)

    return randomIndexes[0: limitRatio], randomIndexes[limitRatio:]


def combineAliAndMisaliVectors(aliV, misaliV, subtomoSize,  shuffle=True):
    """ Generate and subtomos vector and its associated class (aligned or misaligned) from two independent vectors of
    aligned and misaligned subtomos"""

    aliInfoV = np.ones(len(aliV))
    misaliInfoV = np.zeros((len(misaliV)))

    subtomoV = np.concatenate((aliV, misaliV))
    infoV = np.concatenate((aliInfoV, misaliInfoV))

    # # Generate phantom data (Ali as it is, Misali = Ali * -1)
    # aliInfoV = np.ones(len(aliV))
    # misaliInfoV = np.zeros((len(aliV)))
    #
    # subtomoV = np.concatenate((aliV, aliV * -1))
    # infoV = np.concatenate((aliInfoV, misaliInfoV))

    # # Generate phantom data (Ali as it is, Misali = rot90z of ali)
    # aliInfoV = np.ones(len(aliV))
    # misaliInfoV = np.zeros((len(aliV)))
    #
    # subtomoV = np.zeros(((len(aliV)+len(aliV)), *(subtomoSize, subtomoSize, subtomoSize)))
    #
    # for i, _ in enumerate(aliV):
    #     subtomoV[i, :] = aliV[i, :]
    #
    # infoV = np.concatenate((aliInfoV, misaliInfoV))
    #
    # for i, subtomo in enumerate(aliV):
    #     from math import cos, sin
    #     import xmippLib as xmipp
    #
    #     Z_ROTATION_90 = np.asarray([[cos(np.deg2rad(180)), -sin(np.deg2rad(180)), 0],
    #                                 [sin(np.deg2rad(180)), cos(np.deg2rad(180)), 0],
    #                                 [0, 0, 1]])
    #
    #     inputSubtomo = xmipp.Image()
    #     outputSubtomo = np.zeros((len(subtomo), len(subtomo), len(subtomo)))
    #
    #     for slice in range(len(subtomo)):
    #         inputSubtomo.setData(subtomo[slice])
    #
    #         outputSubtomoImage = inputSubtomo.applyWarpAffine(list(Z_ROTATION_90.flatten()),
    #                                                           (len(subtomo), len(subtomo), len(subtomo)),
    #                                                           True)
    #
    #         outputSubtomo[slice] = outputSubtomoImage.getData()
    #
    #     subtomoV[len(aliV)+i, :] = outputSubtomo

    if len(subtomoV) != len(infoV):
        raise Exception("ERROR: len(subtomoV) != len(infoV) " + str(len(subtomoV)) + "!=" + str(len(infoV)))

    if shuffle:
        randomize = np.arange(len(subtomoV))
        np.random.shuffle(randomize)

        subtomoV = subtomoV[randomize]
        infoV = infoV[randomize]

    return subtomoV, infoV
