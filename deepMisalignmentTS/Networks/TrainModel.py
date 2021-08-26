""" This module trains an validate the different models to solve the misalignment detection problem. """

import numpy as np
import os
import sys
from time import time

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from CreateModel import scratchModel
import plotUtils
import utils

batch_size = 128  # Number of boxes per batch


if __name__ == "__main__":

    # Check no program arguments missing
    if len(sys.argv) < 3:
        print("Usage: scipion python batch_deepDefocus.py <stackDir> <modelDir>")
        sys.exit()
    stackDir = sys.argv[1]
    modelDir = sys.argv[2]

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Loading data...")
    start_time = time()
    inputSubtomoStream = np.load(os.path.join(stackDir, "inputDataStream.npy"))
    misalignmentInfoVector = np.load(os.path.join(stackDir, "misalignmentInfoList.npy"))

    # Normalize input subtomo data stream to N(0,1)
    normalizedInputSubtomoStream = utils.normalizeInputDataStream(inputSubtomoStream)

    for i in range(len(misalignmentInfoVector[0, :])):
        # Get statistics
        _, _, _, _, _ = utils.statisticsFromInputDataStream(misalignmentInfoVector, i, verbose=True)

        # Plot variable info histogram
        plotUtils.plotHistogramVariable(misalignmentInfoVector, variable=i)

    # Plot correlation between two variables
    # Centroid X and PCA X
    plotUtils.plotCorrelationVariables(misalignmentInfoVector, variable1=0, variable2=6)
    # Centroid Y and PCA Y
    plotUtils.plotCorrelationVariables(misalignmentInfoVector, variable1=0, variable2=6)

    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

    print("Input subtomograms dimensions: ", np.shape(inputSubtomoStream))
    print("Input misalignment dimensions: ", np.shape(misalignmentInfoVector))

    model = scratchModel
    model.summary()

    print("Train model")
    start_time = time()
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    callbacks_list = [callbacks.CSVLogger("./outCSV_06_28_1", separator=',', append=False),
                      callbacks.TensorBoard(log_dir='./outTB_06_28_1',
                                            histogram_freq=0,
                                            batch_size=128,
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
                                                  min_lr=0)]

    history = model.fit(inputSubtomoStream,
                        misalignmentInfoVector,
                        batch_size=128,
                        epochs=100,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks_list)

    myValLoss = np.zeros(1)
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
    model.save(os.path.join(modelDir, 'model.h5'))
    elapsed_time = time() - start_time
    print("Time spent training the model: %0.10f seconds." % elapsed_time)

    print("Test model")
    start_time = time()
    loadModelDir = os.path.join(modelDir, 'model.txt')
    model = load_model(loadModelDir)
    imagPrediction = model.predict(inputSubtomoStream)
    np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPrediction)
    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds." % elapsed_time)

    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(misalignmentInfoVector, imagPrediction)
    print("Final model mean absolute error val_loss: %f", mae)
