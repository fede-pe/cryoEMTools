
""" Module containing plotting functions of the results and behaviour from the trained networks. """
import numpy as np
from matplotlib import pyplot as plt


def plotHistogramVariable(misalignmentInfoVector, variable):
    """ This method plots and histogram of the selected variable from the input information list.
     Variable indicates the column number of the feature in the information matrix. """

    title = getTitleFromVariable(variable)

    # Histogram plot
    plt.style.use('ggplot')
    plt.subplot(3, 3, (variable+1))
    plt.title(title)
    plt.hist(misalignmentInfoVector[:, variable], bins=50, color='b')

def plotCorrelationVariables(misalignmentInfoVector, variable1, variable2)
    """ This method plots the correlation between the two selected variables from the input information list.
     Variable1 and variable2 indicate the column number of the feature in the information matrix. """

    corr = np.corrcoef(misalignmentInfoVector[:, variable1], misalignmentInfoVector[:, variable2])

    title1 = getTitleFromVariable(variable1)
    title2 = getTitleFromVariable(variable2)

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
