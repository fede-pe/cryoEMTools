#!/usr/bin/env python2
import cv2
import numpy as np
import os
import sys
from time import time
# ----TENSORFLOW INSIDE KERAS
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
#from keras.callbacks import TensorBoard, ModelCheckpoint
#import keras.callbacks as callbacks
#from keras.models import Model
#from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
#from keras.optimizers import Adam
#from keras.models import load_model

batch_size = 3  # 128 should be by default
epochs = 25
training_Bool = False
testing_Bool = True
plots_Bool = True

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ----------- MODEL ARCHITECTURE  -------------------
    def constructModel():
        inputLayer = Input(shape=(512, 512, 3), name="input")
        L = Conv2D(16, (15, 15), activation="relu")(inputLayer)
        L = BatchNormalization()(L)
        L = MaxPooling2D((3, 3))(L)
        L = Conv2D(16, (9, 9), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(16, (5, 5), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)
        L = Flatten()(L)
        L = Dense(4, name="output", activation="linear")(L)
        return Model(inputLayer, L)


    def constructModelBis():
        inputLayer = Input(shape=(512, 512, 3), name="input")
        L = Conv2D(16, (15, 15), activation="relu")(inputLayer)
        L = MaxPooling2D((3, 3))(L)
        L = Conv2D(16, (9, 9), activation="relu")(L)
        L = MaxPooling2D()(L)
        L = Conv2D(32, (5, 5), activation="relu")(L)
        L = MaxPooling2D()(L)
        L = Conv2D(64, (5, 5), activation="relu")(L)
        L = Flatten()(L)
        L = Dense(256, activation="relu")(L)
        L = Dropout(0.2)(L)
        L = Dense(4, name="output", activation="linear")(L)
        return Model(inputLayer, L)

    model = constructModel()
    model.summary()

# ----------- LOADING DATA -------------------
    if len(sys.argv) < 3:
        print("Usage: python3 trainDeepDefocusModel.py <stackDir> <modelDir>")
        sys.exit()
    stackDir = sys.argv[1]
    modelDir = sys.argv[2]

    print("Loading data...")
    imageStackDir = os.path.join(stackDir, "preparedImageStack.npy")
    defocusStackDir = os.path.join(stackDir, "preparedDefocusStack.npy")
    imagMatrix = np.load(imageStackDir)
    defocusVector = np.load(defocusStackDir)

# ----------- TRAINING MODEL-------------------
    if training_Bool:
        print("Train mode")
        start_time = time()
        model = constructModel()
        model.summary()
        optimizer = Adam(lr=0.0001)
        model.compile(loss='mean_absolute_error', optimizer='Adam')
        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

        callbacks_list = [callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1'), separator=',', append=False),
                          callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1'), histogram_freq=0,
                                                batch_size=batch_size, write_graph=True, write_grads=False,
                                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                                embeddings_metadata=None,embeddings_data=None),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)]

        history = model.fit(imagMatrix, defocusVector, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_split=0.1, callbacks=callbacks_list)

        myValLoss = np.zeros(1)
        myValLoss[0] = history.history['val_loss'][-1]
        np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
        model.save(os.path.join(modelDir, 'model.h5'))
        elapsed_time = time() - start_time
        print("Time in training model: %0.10f seconds." % elapsed_time)

# ----------- TESTING MODEL-------------------
    if testing_Bool:
        loadModelDir = os.path.join(modelDir, 'model.h5')
        model = load_model(loadModelDir)
        imagPrediction = model.predict(imagMatrix)
        np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPrediction)

        # DEFOCUS_U
        mae = mean_absolute_error(defocusVector[:, 0], imagPrediction[:, 0])
        print("Defocus U model mean absolute error val_loss: ", mae)

        # DEFOCUS_V
        mae = mean_absolute_error(defocusVector[:, 1], imagPrediction[:, 1])
        print("Defocus V model mean absolute error val_loss: ", mae)

        # DEFOCUS_ANGLE
        mae = mean_absolute_error(defocusVector[:, 2:], imagPrediction[:, 2:])
        print("Defocus angle model mean absolute error val_loss: ", mae)

        #TOTAL ERROR
        mae = mean_absolute_error(defocusVector, imagPrediction)
        print("Final model mean absolute error val_loss: ", mae)

        if plots_Bool:
            plt.plot(defocusVector[:, 0], c='b', label='dU')
            plt.plot(defocusVector[:, 1], c='m', label='dV')
            plt.plot(imagPrediction[:, 0], c='r', label='dU_pred')
            plt.plot(imagPrediction[:, 1], c='y', label='dV_pred')
            plt.legend()
            plt.show()

            plt.plot(defocusVector[:, 2], c='b', label='Sin')
            plt.plot(defocusVector[:, 3], c='m', label='Cos')
            plt.plot(imagPrediction[:, 2], c='r', label='Sin_pred')
            plt.plot(imagPrediction[:, 3], c='y', label='Cos_pred')
            plt.legend()
            plt.show()

    exit(0)
