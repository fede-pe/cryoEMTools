""" This module trains an validate the different models to solve the misalignment detection problem. """
import datetime

import numpy as np
import os
import sys
from time import time

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from CreateModel import compileModel, scratchModel
import plotUtils
import utils

BATCH_SIZE = 128  # Number of boxes per batch
EPOCHS = 10  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate


if __name__ == "__main__":

    # Check no program arguments missing
    if len(sys.argv) < 4:
        print("Usage: scipion python batch_deepDefocus.py <stackDir> <generatePlots 0/1> <verboseOutput 0/1>")
        sys.exit()

    # Path with the input stack of data
    stackDir = sys.argv[1]

    # Generate output plots
    if sys.argv[2] == "0":
        generatePlots = False
    elif sys.argv[2] == "1":
        generatePlots = True
    else:
        raise Exception("Invalid option for <generatePlots 0/1>. This option only accepts 0 or 1 input values.")

    # Verbose output
    if sys.argv[2] == "0":
        verboseOutput = False
    elif sys.argv[2] == "1":
        verboseOutput = True
    else:
        raise Exception("Invalid option for <verboseOutput 0/1>. This option only accepts 0 or 1 input values.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------------------------ PREPROCESS DATA
    print("Loading data...")
    start_time = time()
    inputSubtomoStream = np.load(os.path.join(stackDir, "inputDataStream.npy"))
    misalignmentInfoVector = np.load(os.path.join(stackDir, "misalignmentInfoList.npy"))

    # Normalize input subtomo data stream to N(0,1)
    normalizedInputSubtomoStream = utils.normalizeInputDataStream(inputSubtomoStream)

    # ------------------------------------------------------------ PRODUCE SIDE INFO
    for i in range(len(misalignmentInfoVector[0, :])):
        # Get statistics
        _, _, _, _, _ = utils.statisticsFromInputDataStream(misalignmentInfoVector, i, verbose=verboseOutput)

        # Plot variable info histogram
        pltHist = plotUtils.plotHistogramVariable(misalignmentInfoVector, variable=i)

    if generatePlots:
        pltHist.show()

    # Plot correlation between two variables
    # Centroid X and PCA X
    _ = plotUtils.plotCorrelationVariables(misalignmentInfoVector, variable1=0, variable2=6, counter=1)
    # Centroid Y and PCA Y
    pltCorr = plotUtils.plotCorrelationVariables(misalignmentInfoVector, variable1=1, variable2=7, counter=2)

    if generatePlots:
        pltCorr.show()

    # ------------------------------------------------------------ SPLIT DATA
    normISS_train, normISS_test, misalignmentInfoVector_train, misalignmentInfoVector_test = \
        train_test_split(normalizedInputSubtomoStream, misalignmentInfoVector, test_size=0.15, random_state=42)

    print('Input train matrix: ' + str(np.shape(normISS_train)))
    print('Output train matrix: ' + str(np.shape(misalignmentInfoVector_train)))
    print('Input test matrix: ' + str(np.shape(normISS_test)))
    print('Output test matrix: ' + str(np.shape(misalignmentInfoVector_test)))

    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

    # ------------------------------------------------------------ TRAIN MODEL
    print("Train model")
    start_time = time()

    model = compileModel(model=scratchModel(), learningRate=LEARNING_RATE)

    dateAndTime = str(datetime.datetime.now())

    callbacks_list = [callbacks.CSVLogger("./outCSV_" + dateAndTime + '.log',
                                          separator=',',
                                          append=False),

                      callbacks.TensorBoard(log_dir='./outTB_' + dateAndTime,
                                            histogram_freq=0,
                                            batch_size=BATCH_SIZE,
                                            write_graph=True,
                                            write_grads=False,
                                            write_images=False,
                                            embeddings_freq=0,
                                            embeddings_layer_names=None,
                                            embeddings_metadata=None,
                                            embeddings_data=None),

                      callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.1,
                                                  patience=5,
                                                  verbose=1,
                                                  mode='auto',
                                                  min_delta=0.0001,
                                                  cooldown=0,
                                                  min_lr=0),

                      callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10)
                      ]

    history = model.fit(normISS_train,
                        misalignmentInfoVector_train,
                        batch_size=BATCH_SIZE,
                        epochs=100,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks_list)

    modelDir = os.path.join(stackDir, dateAndTime)

    myValLoss = np.zeros(1)
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)

    model.save(os.path.join(modelDir, 'model.h5'))
    elapsed_time = time() - start_time

    print("Time spent training the model: %0.10f seconds." % elapsed_time)

    plotUtils.plotTraining(history, EPOCHS)

    # ------------------------------------------------------------ TEST MODEL
    print("Test model")
    start_time = time()

    loadModelDir = os.path.join(modelDir, 'model.h5')
    model = load_model(loadModelDir)

    misalignmentInfoVector_prediction = model.predict(normISS_test)
    np.savetxt(os.path.join(stackDir, 'model_prediction.txt'), misalignmentInfoVector_prediction)

    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds." % elapsed_time)

    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_prediction)
    print("Final model mean absolute error val_loss: %f", mae)

    loss = model.evaluate(misalignmentInfoVector_prediction, misalignmentInfoVector_test, verbose=2)
    print("Testing set Total Mean Abs Error: {:5.2f} charges".format(loss))

    for i in range(len(misalignmentInfoVector[0, :])):
        # Plot results from testing
        _, _, _, _, _ = plotUtils.plotTesting(misalignmentInfoVector_test, misalignmentInfoVector_prediction, i)


