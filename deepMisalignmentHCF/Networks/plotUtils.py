""" Module containing plotting functions of the datasets and results. """

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import os


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


def plotClassesDistributionDynamic(aliDict, misaliDict, dirPath):
    """ This method plots and histogram of the classes distributions from each dataset between aligned and misaligned
    subtomos. """

    aliSubtomos = []
    misaliSubtomos = []
    populations = []

    unionKeys = set(aliDict.keys()) | set(misaliDict.keys())

    for key in unionKeys:
        populations.append(key)

        if key in aliDict.keys():
            aliSubtomos.append(aliDict[key][1])
        else:
            aliSubtomos.append(0)

        if key in misaliDict.keys():
            misaliSubtomos.append(misaliDict[key][1])
        else:
            misaliSubtomos.append(0)

    # Set up positions for bars on X-axis
    bar_width = 0.35
    index = np.arange(len(populations))

    # Create bar plot
    fig, ax = plt.subplots()
    _ = ax.bar(index, aliSubtomos, bar_width, label='AliSubtomos')
    _ = ax.bar(index + bar_width, misaliSubtomos, bar_width, label='MisaliSubtomos')

    # Customize the plot
    ax.set_xlabel('Population')
    ax.set_ylabel('Number of subtomos')
    ax.set_title('Classes distribution')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(populations)
    ax.legend()

    # Display the plot
    plt.savefig(os.path.join(dirPath, "classesDistribution.png"))
    plt.close()


def plotTraining(history, epochs, dirPath):
    """ This method generates training post from the history of the model."""

    # Loss plot
    plt.title('Loss')

    plt.plot(history.history['loss'], 'b', label='training loss')
    plt.plot(history.history['val_loss'], 'r', label='validation loss')

    plt.xlabel("Epochs")
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig(os.path.join(dirPath, "loss.png"))
    plt.close()

    plt.tick_params('y', colors='b')
    plt.gca().set_xlim(0, epochs - 1)
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(history.epoch, history.history["val_loss"], "r^-")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Reduce LR on Plateau", fontsize=14)
    plt.savefig(os.path.join(dirPath, "learningRate.png"))
    plt.close()


def plotTesting(misalignmentInfoVector_test, misalignmentInfoVector_prediction, dirPath):
    """ This method generates testing post from the history of the model.
    Variable indicates the column number of the feature in the information matrix."""

    title = "Prediction confusion matrix"
    classes_pred = ["Misaligned_pred", "Aligned_pred"]
    classes_gt = ["Misaligned_gt", "Aligned_gt"]

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
                         index=[i for i in classes_pred],
                         columns=[i for i in classes_gt])

    plt.figure(figsize=(10, 7))
    plt.title(title)

    sn.heatmap(df_cm, annot=True)

    plt.savefig(os.path.join(dirPath, "confussionMatrix.png"))
    plt.close()
