""" This module trains an validate the different models to solve the misalignment detection problem. """
import datetime

import numpy as np
import os
import sys
from time import time

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from CreateModel import compileModel, scratchModel
from classes import DataGenerator
import plotUtils
import utils

SUBTOMO_SIZE = 32  # Dimensions of the subtomos (cubic, SUBTOMO_SIZE x SUBTOMO_SIZE x SUBTOMO_SIZE shape)
BATCH_SIZE = 128  # Number of boxes per batch
EPOCHS = 2  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate
VALIDATION_SPLIT = 0.2  # Ratio of data used for validation

if __name__ == "__main__":

    # Check no program arguments missing
    if len(sys.argv) < 4:
        print("Usage: scipion python batch_deepDefocus.py <stackDir> <verboseOutput 0/1> <generatePlots 0/1>")
        sys.exit()

    # Path with the input stack of data
    stackDir = sys.argv[1]

    # Verbose output
    if sys.argv[2] == "0":
        verboseOutput = False
    elif sys.argv[2] == "1":
        verboseOutput = True
    else:
        raise Exception("Invalid option for <verboseOutput 0/1>. This option only accepts 0 or 1 input values.")

    # Generate output plots
    if sys.argv[3] == "0":
        generatePlots = False
    elif sys.argv[3] == "1":
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

    # ------------------------------------------------------------ PRODUCE SIDE INFO
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
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

    normalizedInputSubtomoStreamAli = np.concatenate((normalizedInputSubtomoStreamAli,
                                                      generatedSubtomosArrayAli))

    print("Data structure shapes AFTER augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

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

    # ------------------------------------------------------------ SPLIT DATA
    # Aligned subtomos
    normISSAli_train, normISSAli_test = train_test_split(normalizedInputSubtomoStreamAli,
                                                         test_size=0.15,
                                                         random_state=42)

    # Misligned subtomos
    normISSMisali_train, normISSMisali_test = train_test_split(normalizedInputSubtomoStreamMisali,
                                                               test_size=0.15,
                                                               random_state=42)

    print("Data objects final dimensions")
    print('Input train aligned subtomos matrix: ' + str(np.shape(normISSAli_train)))
    print('Input test aligned subtomos matrix: ' + str(np.shape(normISSAli_test)))
    print('Input train misaligned subtomos matrix: ' + str(np.shape(normISSMisali_train)))
    print('Input test misaligned subtomos matrix: ' + str(np.shape(normISSMisali_test)) + '\n')

    elapsed_time = time() - start_time
    print("Overall time spent preparing the data: %0.10f seconds.\n\n" % elapsed_time)

    # ------------------------------------------------------------ TRAIN MODEL
    print("Train model")
    start_time = time()

    # Validation/training ID toggle vectors
    aliID = utils.generateTraningValidationVectors(len(normISSAli_train), VALIDATION_SPLIT)
    misaliID = utils.generateTraningValidationVectors(len(normISSMisali_train), VALIDATION_SPLIT)

    # Parameters
    params = {'dim': (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE),
              'batch_size': BATCH_SIZE,
              'n_classes': 6,
              'shuffle': True}

    # Generators
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)

    # Design model
    model = compileModel(model=scratchModel(), learningRate=LEARNING_RATE)

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  workers=6)

    myValLoss = np.zeros(1)
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.txt'), myValLoss)

    model.save(os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.h5'))
    elapsed_time = time() - start_time

    print("Time spent training the model: %0.10f seconds." % elapsed_time)

    if generatePlots:
        plotUtils.plotTraining(history, EPOCHS)

    # ------------------------------------------------------------ TEST MODEL
    print("\n\nTest model...\n")
    start_time = time()

    loadModelDir = os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.h5')
    model = load_model(loadModelDir)

    misalignmentInfoVector_prediction = model.predict(normISS_test)

    # Convert the set of probabilities from the previous command into the set of predicted classes
    misalignmentInfoVector_predictionClasses = np.argmax(misalignmentInfoVector_prediction, axis=1)

    np.savetxt(os.path.join(stackDir, 'model_prediction.txt'),
               misalignmentInfoVector_predictionClasses)

    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)

    mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
    print("Final model mean absolute error val_loss: %f\n" % mae)

    loss = model.evaluate(normISS_test,
                          misalignmentInfoVector_test,
                          verbose=2)
    print("\nTesting set total mean absolute error: {:5.2f}\n".format(loss))

    # Plot results from testing
    plotUtils.plotTesting(
        misalignmentInfoVector_test,
        misalignmentInfoVector_predictionClasses
    )
