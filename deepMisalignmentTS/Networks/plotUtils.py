
""" Module containing plotting functions of the datasets and results. """

import numpy as np
from matplotlib import pyplot as plt
import utils


def plotHistogramVariable(misalignmentInfoVector, variable):
    """ This method plots and histogram of the selected variable from the input information list.
     Variable indicates the column number of the feature in the information matrix. """

    title = utils.getTitleFromVariable(variable)

    # Histogram plot
    plt.style.use('ggplot')
    plt.subplot(3, 3, (variable+1))
    plt.title(title)
    plt.hist(misalignmentInfoVector[:, variable], bins=50, color='b')


def plotCorrelationVariables(misalignmentInfoVector, variable1, variable2):
    """ This method plots the correlation between the two selected variables from the input information list.
     Variable1 and variable2 indicate the column number of the feature in the information matrix. """

    corr = np.corrcoef(misalignmentInfoVector[:, variable1], misalignmentInfoVector[:, variable2])

    title1 = utils.getTitleFromVariable(variable1)
    title2 = utils.getTitleFromVariable(variable2)

    # Correlation plot
    plt.title('Pearson correlation = ' + "{:.5f}".format(corr[0, 1]))
    plt.scatter(misalignmentInfoVector[:, variable1], misalignmentInfoVector[:, variable2])
    plt.xlabel(title1)
    plt.ylabel(title2)
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()
