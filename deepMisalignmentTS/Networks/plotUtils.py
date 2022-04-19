""" Module containing plotting functions of the datasets and results. """

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


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


def plotClassesDistributionDynamic(numberOfAlignedSubtomos, numberOfMisalignedSubtomos):
    """ This method plots and histogram of the classes distributions from the dataset between aligned and misaligned
    subtomos in dynamic training mode. """

    title = "Classes distribution"

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


def plotTesting(misalignmentInfoVector_prediction, misalignmentInfoVector_test):
    """ This method generates testing post from the history of the model.
    Variable indicates the column number of the feature in the information matrix."""

    title = "Prediction confusion matrix"
    classes = ["Misaligned", "Aligned"]

    confusionMatrix = np.zeros((2, 2))

    for i in range(len(misalignmentInfoVector_prediction)):
        prediction = misalignmentInfoVector_prediction[i]
        test = misalignmentInfoVector_test[i]

        # True negative
        if prediction == 0 and test == 0:
            confusionMatrix[0][0] += 1

        # False negative
        elif prediction == 0 and test == 1:
            confusionMatrix[0][1] += 1

        # True positive
        elif prediction == 1 and test == 1:
            confusionMatrix[1][1] += 1

        # False positive
        elif prediction == 1 and test == 0:
            confusionMatrix[1][0] += 1

    df_cm = pd.DataFrame(confusionMatrix,
                         index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(10, 7))
    plt.title(title)

    sn.heatmap(df_cm, annot=True)

    plt.show()
