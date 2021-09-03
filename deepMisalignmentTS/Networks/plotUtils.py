""" Module containing plotting functions of the datasets and results. """

from matplotlib import pyplot as plt
import numpy as np


def plotClassesDistribution(misalignmentInfoVector):
    """ This method plots and histogram of the classes distributions from the dataset between aligned and misaligned
    subtomos. """

    title = "Classes distribution"

    numberOfMisalignedSubtomos = 0
    numberOfAlignedSubtomos = 0

    for subtomo in misalignmentInfoVector:
        if subtomo == 0:
            numberOfMisalignedSubtomos += 1
        elif subtomo == 1:
            numberOfAlignedSubtomos += 1

    classes = ["Aligned", "Misaligned"]
    classesHeight = [numberOfAlignedSubtomos, numberOfMisalignedSubtomos]

    # Histogram plot
    plt.style.use('ggplot')

    plt.title(title)

    plt.bar(classes, classesHeight, color='r')

    plt.show()


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


def plotTesting(misalignmentInfoVector_prediction, misalignmentInfoVector_test, model):
    """ This method generates testing post from the history of the model.
    Variable indicates the column number of the feature in the information matrix."""

    title = "Prediction confusion matrix"

    # confusionMatrix = np.zeros((2, 2))
    #
    # for i in range(misalignmentInfoVector_prediction):
    #     prediction = misalignmentInfoVector_prediction[i]
    #     test = misalignmentInfoVector_test[i]
    #
    #     # True negative
    #     if prediction == 0 and test == 0:
    #         confusionMatrix[0][0] += 1
    #
    #     # False negative
    #     elif prediction == 0 and test == 1:
    #         confusionMatrix[0][1] += 1
    #
    #     # True positive
    #     elif prediction == 1 and test == 1:
    #         confusionMatrix[1][1] += 1
    #
    #     # False positive
    #     elif prediction == 0 and test == 1:
    #         confusionMatrix[0][1] += 1

    from sklearn.metrics import plot_confusion_matrix

    disp = plot_confusion_matrix(model,
                                 misalignmentInfoVector_test,
                                 misalignmentInfoVector_prediction)

    disp.ax_.set_title(title)

    plt.title(title)

    plt.scatter(x, misalignmentInfoVector_test[:, variable], c='r', label=title)
    plt.scatter(x, misalignmentInfoVector_prediction[:, variable], c='b', label=title + "_pred")

    plt.legend()
    plt.show()
