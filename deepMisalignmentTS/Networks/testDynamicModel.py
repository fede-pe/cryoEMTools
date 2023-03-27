""" This module trains and validate the different models to solve the misalignment detection problem. """


import datetime
from time import time
import os
import sys

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import numpy as np

from CreateModel import compileModel, scratchModel, getCallbacks
from classes import DataGenerator
import plotUtils
import utils

SUBTOMO_SIZE = 32  # Dimensions of the subtomos (cubic, SUBTOMO_SIZE x SUBTOMO_SIZE x SUBTOMO_SIZE shape)
BATCH_SIZE = 64  # Number of boxes per batch
NUMBER_RANDOM_BATCHES = -1

if __name__ == "__main__":
    # Running command

    # Check no program arguments missing
    if len(sys.argv) == 5:
        retrainModel = True
        print("Start testing model mode")

    else:
        print("Usage: python3 testDynamicModel.py <stackDir> <modelDir> <verboseOutput 0/1> <generatePlots 0/1>")
        sys.exit()

    # Path for the input stack of data
    stackDir = sys.argv[1]

    # Path for the testing model
    modelDir = sys.argv[2]

    # Verbose output
    if sys.argv[3] == "0":
        verboseOutput = False
    elif sys.argv[3] == "1":
        verboseOutput = True
    else:
        raise Exception("Invalid option for <verboseOutput 0/1>. This option only accepts 0 or 1 input values.")

    # Generate output plots
    if sys.argv[4] == "0":
        generatePlots = False
    elif sys.argv[4] == "1":
        generatePlots = True
    else:
        raise Exception("Invalid option for <generatePlots 0/1>. This option only accepts 0 or 1 input values.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------------------------ PREPROCESS DATA
    print("Loading data...")
    start_time = time()
    inputSubtomoStreamAli = np.load(os.path.join(stackDir, "inputDataStreamAli.npy"))
    inputSubtomoStreamMisali = np.load(os.path.join(stackDir, "inputDataStreamMisali.npy"))

    numberOfAliSubtomos = len(inputSubtomoStreamAli)
    numberOfMisaliSubtomos = len(inputSubtomoStreamMisali)
    totalSubtomos = numberOfAliSubtomos + numberOfMisaliSubtomos

    # Normalize input subtomo data stream to N(0,1)
    normalizedInputSubtomoStreamAli = utils.normalizeInputDataStream(inputSubtomoStreamAli)
    normalizedInputSubtomoStreamMisali = utils.normalizeInputDataStream(inputSubtomoStreamMisali)

    # Test normalization
    # print("\n")
    # print("normalizedInputSubtomoStreamAli stats: mean " + str(np.mean(normalizedInputSubtomoStreamAli)) +
    #       " std: " + str(np.std(normalizedInputSubtomoStreamAli)))
    # print("normalizedInputSubtomoStreamMisali stats: mean " + str(np.mean(normalizedInputSubtomoStreamMisali)) +
    #       " std: " + str(np.std(normalizedInputSubtomoStreamMisali)))
    # print("\n")

    # ------------------------------------------------------------ PRODUCE SIDE INFO
    # Update the number of random batches respect to the dataset and batch sizes
    NUMBER_RANDOM_BATCHES = totalSubtomos // BATCH_SIZE

    # Output classes distribution info
    aliSubtomosRatio = numberOfAliSubtomos / totalSubtomos
    misaliSubtomosRatio = numberOfMisaliSubtomos / totalSubtomos

    if verboseOutput:
        print("\nClasses distribution:\n"
              "Aligned: %d (%.3f%%)\n"
              "Misaligned: %d (%.3f%%)\n\n"
              % (numberOfAliSubtomos, aliSubtomosRatio * 100,
                 numberOfMisaliSubtomos, misaliSubtomosRatio * 100))

    # Plot classes distribution info histogram
    if generatePlots:
        plotUtils.plotClassesDistributionDynamic(numberOfAliSubtomos, numberOfMisaliSubtomos)

    # ------------------------------------------------------------ DATA AUGMENTATION
    """
    start_time = time()

    # Data augmentation for aligned subtomos
    generatedSubtomosAli = []

    print("Data augmentation for aligned subtomos:")

    for i in range(numberOfAliSubtomos):

        newSubtomos = utils.dataAugmentationSubtomoDynamic(normalizedInputSubtomoStreamAli[i, :, :, :],
                                                           (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE))

        for subtomo in newSubtomos:
            generatedSubtomosAli.append(subtomo)

    generatedSubtomosArrayAli = np.zeros((len(generatedSubtomosAli),
                                          SUBTOMO_SIZE,
                                          SUBTOMO_SIZE,
                                          SUBTOMO_SIZE),
                                         dtype=np.float64)

    for i, subtomo in enumerate(generatedSubtomosAli):
        generatedSubtomosArrayAli[i, :, :, :] = subtomo[:, :, :]

    print("Number of new subtomos generated: %d\n" % len(generatedSubtomosAli))

    print("Data structure shapes BEFORE augmentation:")
    print("Input aligned subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

    normalizedInputSubtomoStreamAli = np.concatenate((normalizedInputSubtomoStreamAli,
                                                      generatedSubtomosArrayAli))

    print("Data structure shapes AFTER augmentation:")
    print("Input misaligned subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

    # Data augmentation for misaligned subtomos
    generatedSubtomosMisali = []

    print("Data augmentation for misaligned subtomos:")

    for i in range(numberOfMisaliSubtomos):

        newSubtomos = utils.dataAugmentationSubtomoDynamic(normalizedInputSubtomoStreamMisali[i, :, :, :],
                                                           (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE))

        for subtomo in newSubtomos:
            generatedSubtomosMisali.append(subtomo)

    generatedSubtomosArrayMisali = np.zeros((len(generatedSubtomosMisali),
                                             SUBTOMO_SIZE,
                                             SUBTOMO_SIZE,
                                             SUBTOMO_SIZE),
                                            dtype=np.float64)

    for i, subtomo in enumerate(generatedSubtomosMisali):
        generatedSubtomosArrayMisali[i, :, :, :] = subtomo[:, :, :]

    print("Number of new subtomos generated: %d\n" % len(generatedSubtomosMisali))

    print("Data structure shapes BEFORE augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamMisali.shape))

    normalizedInputSubtomoStreamMisali = np.concatenate((normalizedInputSubtomoStreamMisali,
                                                         generatedSubtomosArrayMisali))

    print("Data structure shapes AFTER augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamMisali.shape))

    dataAug_time = time() - start_time
    print("Time spent in data augmentation: %0.10f seconds.\n\n" % dataAug_time)
    """
    # ------------------------------------------------------------ TEST MODEL
    print("\n\nTest model...\n")
    start_time = time()

    model = load_model(modelDir)

    normISS_test, misalignmentInfoVector_test = utils.combineAliAndMisaliVectors(normalizedInputSubtomoStreamAli,
                                                                                 normalizedInputSubtomoStreamMisali,
                                                                                 SUBTOMO_SIZE,
                                                                                 shuffle=False)

    misalignmentInfoVector_prediction = model.predict(normISS_test)

    # print("misalignmentInfoVector_prediction")
    # print(misalignmentInfoVector_prediction)
    # print("misalignmentInfoVector_test")
    # print(misalignmentInfoVector_test)

    # Convert the set of probabilities from the previous command into the set of predicted classes
    misalignmentInfoVector_predictionClasses = np.round(misalignmentInfoVector_prediction)

    print(misalignmentInfoVector_predictionClasses)

    np.savetxt(os.path.join(stackDir, 'model_prediction.txt'),
               misalignmentInfoVector_predictionClasses)

    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)

    mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
    print("Final model mean absolute error val_loss: %f\n" % mae)

    loss = model.evaluate(normISS_test,
                          misalignmentInfoVector_test,
                          verbose=2)

    print("Testing set mean absolute error: {:5.4f}".format(loss[0]))
    print("Testing set accuracy: {:5.4f}".format(loss[1]))

    # Plot results from testing
    plotUtils.plotTesting(
        misalignmentInfoVector_test,
        misalignmentInfoVector_predictionClasses
    )
