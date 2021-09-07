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
import plotUtils
import utils

SUBTOMO_SIZE = 32  # Dimensions of the subtomos (cubic, SUBTOMO_SIZE x SUBTOMO_SIZE x SUBTOMO_SIZE shape)
BATCH_SIZE = 128  # Number of boxes per batch
EPOCHS = 2  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate

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
    inputSubtomoStream = np.load(os.path.join(stackDir, "inputDataStream.npy"))
    misalignmentInfoVector = np.load(os.path.join(stackDir, "misalignmentInfoList.npy"))

    # Normalize input subtomo data stream to N(0,1)
    normalizedInputSubtomoStream = utils.normalizeInputDataStream(inputSubtomoStream)

    # ------------------------------------------------------------ PRODUCE SIDE INFO
    # Output classes distribution info
    if verboseOutput:
        alignedSubtomosRatio = utils.produceClassesDistributionInfo(misalignmentInfoVector, True)
    else:
        alignedSubtomosRatio = utils.produceClassesDistributionInfo(misalignmentInfoVector, False)

    # Plot classes distribution info histogram
    if generatePlots:
        plotUtils.plotClassesDistribution(misalignmentInfoVector)

    # ------------------------------------------------------------ DATA AUGMENTATION
    # Augmentation ratio of the input data
    foldAugmentation = int(0.5 / alignedSubtomosRatio)

    generatedSubtomos = []
    generatedMisalignment = []

    for i in range(normalizedInputSubtomoStream.shape[0]):
        alignment = misalignmentInfoVector[i]

        # Use data augmentation only for properly aligned subtomos (the least numbered group in the dataset)
        if alignment == 1:
            newSubtomos, newMisalignment = utils.dataAugmentationSubtomo(normalizedInputSubtomoStream[i, :, :, :],
                                                                         alignment,
                                                                         foldAugmentation - 1,
                                                                         (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE))

            # generatedSubtomos.append(newSubtomos)

            for subtomo in newSubtomos:
                generatedSubtomos.append(subtomo)

            for misalignment in newMisalignment:
                generatedMisalignment.append(misalignment)

    generatedSubtomosArray = np.zeros((len(generatedSubtomos),
                                       SUBTOMO_SIZE,
                                       SUBTOMO_SIZE,
                                       SUBTOMO_SIZE),
                                      dtype=np.float64)

    for i, subtomo in enumerate(generatedSubtomos):
        generatedSubtomosArray[i, :, :, :] = subtomo[:, :, :]

    generatedMisalignmentArray = np.zeros((len(generatedMisalignment)))

    for i, misalignment in enumerate(generatedMisalignment):
        generatedMisalignmentArray[i] = misalignment

    print("Data augmentation:")
    print("Number of new subtomos generated: %d\n" % len(generatedSubtomos))

    print("Data structure shapes BEFORE augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStream.shape))
    print("Input misalignment info vector: " + str(misalignmentInfoVector.shape) + "\n")

    normalizedInputSubtomoStream = np.concatenate((normalizedInputSubtomoStream, generatedSubtomosArray))
    misalignmentInfoVector = np.concatenate((misalignmentInfoVector, generatedMisalignment))

    print("Data structure shapes AFTER augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStream.shape))
    print("Input misalignment info vector: " + str(misalignmentInfoVector.shape) + "\n")

    # ------------------------------------------------------------ SPLIT DATA
    normISS_train, normISS_test, misalignmentInfoVector_train, misalignmentInfoVector_test = \
        train_test_split(normalizedInputSubtomoStream, misalignmentInfoVector, test_size=0.15, random_state=42)

    print("Data objects final dimensions")
    print('Input train matrix: ' + str(np.shape(normISS_train)))
    print('Output train matrix: ' + str(np.shape(misalignmentInfoVector_train)))
    print('Input test matrix: ' + str(np.shape(normISS_test)))
    print('Output test matrix: ' + str(np.shape(misalignmentInfoVector_test)) + '\n\n')

    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds.\n\n" % elapsed_time)

    # ------------------------------------------------------------ TRAIN MODEL
    print("Train model")
    start_time = time()

    model = compileModel(model=scratchModel(), learningRate=LEARNING_RATE)

    dateAndTime = str(datetime.datetime.now())
    dateAndTimeVector = dateAndTime.split(' ')
    dateAndTime = dateAndTimeVector[0] + "_" + dateAndTimeVector[1]

    callbacks_list = [
        callbacks.CSVLogger(
            os.path.join(stackDir, "outputLog_" + dateAndTime + "/outCSV_" + dateAndTime + '.log'),
            separator=',',
            append=False
        ),

        callbacks.TensorBoard(
            log_dir=os.path.join(stackDir, "outputLog_" + dateAndTime + "/outTB_" + dateAndTime),
            histogram_freq=0,
            batch_size=BATCH_SIZE,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None
        ),

        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        ),

        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        )
    ]

    history = model.fit(normISS_train,
                        misalignmentInfoVector_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks_list)

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
