
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

    return plt


def plotCorrelationVariables(misalignmentInfoVector, variable1, variable2, counter):
    """ This method plots the correlation between the two selected variables from the input information list.
     Variable1 and variable2 indicate the column number of the feature in the information matrix. """

    corr = np.corrcoef(misalignmentInfoVector[:, variable1], misalignmentInfoVector[:, variable2])

    title1 = utils.getTitleFromVariable(variable1)
    title2 = utils.getTitleFromVariable(variable2)

    # Correlation plot
    plt.subplot(2, 1, counter)

    plt.title('Pearson correlation = ' + "{:.5f}".format(corr[0, 1]))
    plt.scatter(misalignmentInfoVector[:, variable1], misalignmentInfoVector[:, variable2])

    plt.xlabel(title1)
    plt.ylabel(title2)

    plt.axis('equal')

    return plt


def plotTraining(history, epochs):
    """ This method generates training post from the history of the model."""

    # Loss plot
    plt.title('Loss')

    plt.plot(history.history['loss'], 'b', label='training loss')
    plt.plot(history.history['val_loss'], 'r', label='validation loss')

    plt.xlabel("Epochs")
    plt.ylabel('Loss')

    plt.legend()
    plt.show()

    # Learning rate plot
    plt.plot(history.epoch, history.history["lr"], "bo-")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate", color='b')

    plt.tick_params('y', colors='b')
    plt.gca().set_xlim(0, epochs - 1)
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(history.epoch, history.history["val_loss"], "r^-")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Reduce LR on Plateau", fontsize=14)
    plt.show()


def plotTesting(misalignmentInfoVector_test, misalignmentInfoVector_prediction, variable):
    """ This method generates testing post from the history of the model.
    Variable indicates the column number of the feature in the information matrix."""

    title = utils.getTitleFromVariable(variable)

    x = range(1, len(misalignmentInfoVector_test[:, variable]) + 1)

    plt.subplot(3, 3, (variable+1))
    plt.title('Defocus U')

    plt.scatter(x, misalignmentInfoVector_test[:, variable], c='r', label=title)
    plt.scatter(x, misalignmentInfoVector_prediction[:, variable], c='b', label=title + "_pred")

    plt.legend()
    plt.show()
