
""" Module containing functions to parse and process datasets and results. """

import numpy as np


def normalizeInputDataStream(inputSubtomoStream):
    """ Method to normalize the input subtomo data stream to """
    std = inputSubtomoStream.std()
    mean = inputSubtomoStream.mean()

    normalizedInputDataStream = (inputSubtomoStream - mean) / std

    return normalizedInputDataStream


def statisticsFromInputDataStream(misalignmentInfoVector, variable, verbose=False):
    """ This method calculates and outputs the statistics of the selected variable from the input information list.
     Variable indicates the column number of the feature in the information matrix. """

    mean = misalignmentInfoVector[:, variable].mean()
    std = misalignmentInfoVector[:, variable].std()
    median = np.median(misalignmentInfoVector[:, variable])
    min = misalignmentInfoVector[:, variable].min()
    max = misalignmentInfoVector[:, variable].max()

    if verbose:
        title = getTitleFromVariable(variable)
        title = "------------------ " + title + " statistics:"

        print(title)

        print('Mean: ' + str(mean))
        print('Std: ' + str(std))
        print('Median: ' + str(median))
        print('Min: ' + str(min))
        print('Max: ' + str(max))
        print('\n')

    return mean, std, median, min, max


def getTitleFromVariable(variable):
    """ This method returns the title (variable name) from the input variable identifier. """

    if variable == 0:
        title = "Centroid X"
    elif variable == 1:
        title = "Centroid Y"
    elif variable == 2:
        title = "Max distance"
    elif variable == 3:
        title = "Total distance"
    elif variable == 4:
        title = "Hull area"
    elif variable == 5:
        title = "Hull perimeter"
    elif variable == 6:
        title = "PCA X"
    elif variable == 7:
        title = "PCA Y"
    else:
        raise Exception("Variable %d code is out of range" % variable)

    return title

