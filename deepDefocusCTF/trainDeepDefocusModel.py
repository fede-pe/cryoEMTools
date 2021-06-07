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
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from keras.callbacks import TensorBoard, ModelCheckpoint
#import keras.callbacks as callbacks
#from keras.models import Model
#from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
#from keras.optimizers import Adam
#from keras.models import load_model

BATCH_SIZE = 128  # 128 should be by default (The higher the faster it converge)
EPOCHS = 100
LEARNING_RATE = 0.001
training_Bool = True
testing_Bool = True
plots_Bool = True
TEST_SIZE = 0.2


# ------------------------ MAIN PROGRAM -----------------------------

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------- UTILS METHODS --------------------------------------
    def make_training_plots(history):
        # plot loss during training to CHECK OVERFITTING
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(history.history['loss'], 'b+', label='training loss')
        plt.plot(history.history['val_loss'], 'r+', label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.legend()
        # plot mae during training
        plt.subplot(212)
        plt.title('MAE')
        plt.plot(history.history['mae'], label='train')
        plt.plot(history.history['val_mse'], label='validation')
        plt.xlabel("Epochs")
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

        # Plot Learning Rate decreasing
        plt.plot(history.epoch, history.history["lr"], "bo-")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate", color='b')
        plt.tick_params('y', colors='b')
        plt.gca().set_xlim(0, EPOCHS - 1)
        plt.grid(True)
        ax2 = plt.gca().twinx()
        ax2.plot(history.epoch, history.history["val_loss"], "r^-")
        ax2.set_ylabel('Validation Loss', color='r')
        ax2.tick_params('y', colors='r')
        plt.title("Reduce LR on Plateau", fontsize=14)
        plt.show()


    def make_testing_plots(imagPrediction, defocusVector):
        # DEFOCUS PLOT
        x = range(1, len(defocusVector[:, 0])+1)
        print(len(x))
        print(len(defocusVector[:, 0]))
        plt.subplot(211)
        plt.title('Defocus U')
        plt.scatter(x, defocusVector[:, 0], c='r', label='dU')
        plt.scatter(x, imagPrediction[:, 0], c='b', label='dU_pred')
        plt.subplot(212)
        plt.title('Defocus V')
        plt.scatter(x, defocusVector[:, 1], c='r', label='dV')
        plt.scatter(x, imagPrediction[:, 1], c='b', label='dV_pred')
        plt.legend()
        plt.show()

        # DEFOCUS ANGLE PLOT
        plt.subplot(211)
        plt.title('Sin(2*angle)')
        plt.scatter(x, defocusVector[:, 2], c='r', label='Sin')
        plt.scatter(x, imagPrediction[:, 2], c='b', label='Sin_pred')
        plt.subplot(212)
        plt.title('Cos(2*angle)')
        plt.scatter(x, defocusVector[:, 3], c='r', label='Cos')
        plt.scatter(x, imagPrediction[:, 3], c='b', label='Cos_pred')
        plt.legend()
        plt.show()

        # DEFOCUS PREDICTED VS REAL
        plt.subplot(211)
        plt.title('Defocus U')
        plt.scatter(defocusVector[:, 0], imagPrediction[:, 0])
        plt.xlabel('True Values [defocus U]')
        plt.ylabel('Predictions [defocus U]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.subplot(212)
        plt.title('Defocus V')
        plt.scatter(defocusVector[:, 1], imagPrediction[:, 1])
        plt.xlabel('True Values [defocus V]')
        plt.ylabel('Predictions [defocus v]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()

        # DEFOCUS ANGLE PREDICTED VS REAL !OJO NO VA MUY BIEN ESTE PLOT
        plt.subplot(211)
        plt.title('Sin ( 2 * angle)')
        plt.scatter(defocusVector[:, 2], imagPrediction[:, 2])
        plt.xlabel('True Values [Sin]')
        plt.ylabel('Predictions [Sin]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.subplot(212)
        plt.title('Cos (2 * angle)')
        plt.scatter(defocusVector[:, 3], imagPrediction[:, 3])
        plt.xlabel('True Values [Cos]')
        plt.ylabel('Predictions [Cos]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()

        # DEFOCUS ERROR
        plt.subplot(211)
        plt.title('Defocus U')
        error = imagPrediction[:, 0] - defocusVector[:, 0]
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error Defocus U")
        _ = plt.ylabel("Count")
        plt.subplot(212)
        plt.title('Defocus V')
        error = imagPrediction[:, 1] - defocusVector[:, 1]
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error Defocus V")
        plt.show()

        # DEFOCUS ANGLE ERROR
        plt.subplot(211)
        plt.title('Sin(2*Angle)')
        error = imagPrediction[:, 2] - defocusVector[:, 2]
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error")
        _ = plt.ylabel("Count")
        plt.subplot(212)
        plt.title('Cos(2*Angle)')
        error = imagPrediction[:, 3] - defocusVector[:, 3]
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error")
        plt.show()


# ----------- MODEL ARCHITECTURE  -------------------
    def constructModel():
        inputLayer = Input(shape=(512, 512, 3), name="input")
        L = Conv2D(16, (15, 15), activation="relu")(inputLayer)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
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

        model = Model(inputLayer, L)
        model.summary()
        optimizer = Adam(lr=LEARNING_RATE)  # SGD(lr=LEARNING_RATE, momentum=0.9) this could be used to have less EPOCHS but slow the training
        model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae', 'mse'])  #MAE is more robust to outliers

        return model


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

    # split into train and test
    n = len(defocusVector)
    imagMatrix_train, imagMatrix_test = imagMatrix[:int(n*(1-TEST_SIZE)), :, :, :],  imagMatrix[int(n*(1-TEST_SIZE)):, :, :, :]
    defocusVector_train, defocusVector_test = defocusVector[:int(n*(1-TEST_SIZE)), :], defocusVector[int(n*(1-TEST_SIZE)):, :]

    print(np.shape(imagMatrix_train))
    print(np.shape(imagMatrix_test))
    print(np.shape(defocusVector_train))
    print(np.shape(defocusVector_test))


# ----------- TRAINING MODEL-------------------
    if training_Bool:
        print("Train mode")
        start_time = time()
        model = constructModel()

        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

        callbacks_list = [callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1'), separator=',', append=False),
                          callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1'), histogram_freq=0,
                                                batch_size=BATCH_SIZE, write_graph=True, write_grads=False,
                                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                                embeddings_metadata=None, embeddings_data=None),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0),  # reduce lr when a metric has stopped improving dynamically
                          callbacks.EarlyStopping(monitor='val_loss', patience=10)  # The patience parameter is the amount of epochs to check for improvement
                          ]


        history = model.fit(imagMatrix_train, defocusVector_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose='auto',
                            validation_split=0.1, callbacks=callbacks_list)

        myValLoss = np.zeros(1)
        myValLoss[0] = history.history['val_loss'][-1]
        np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
        model.save(os.path.join(modelDir, 'model.h5'))
        elapsed_time = time() - start_time
        print("Time in training model: %0.10f seconds." % elapsed_time)

        if plots_Bool:
            make_training_plots(history)



# ----------- TESTING MODEL-------------------
    if testing_Bool:
        loadModelDir = os.path.join(modelDir, 'model.h5')
        model = load_model(loadModelDir)
        imagPrediction = model.predict(imagMatrix_test)
        np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPrediction)

        print(np.shape(imagPrediction))
        print(np.shape(defocusVector_test))

        # DEFOCUS_U
        mae = mean_absolute_error(defocusVector_test[:, 0], imagPrediction[:, 0])
        print("Defocus U model mean absolute error val_loss: ", mae)

        # DEFOCUS_V
        mae = mean_absolute_error(defocusVector_test[:, 1], imagPrediction[:, 1])
        print("Defocus V model mean absolute error val_loss: ", mae)

        # DEFOCUS_ANGLE
        mae = mean_absolute_error(defocusVector_test[:, 2:], imagPrediction[:, 2:])
        print("Defocus angle model mean absolute error val_loss: ", mae)

        # TOTAL ERROR
        mae = mean_absolute_error(defocusVector_test, imagPrediction)
        print("Final model mean absolute error val_loss: ", mae)

        loss, mae, mse = model.evaluate(imagMatrix_test, defocusVector_test, verbose=2)
        print("Testing set Mean Abs Error: {:5.2f} charges".format(mae))

        if plots_Bool:
            make_testing_plots(imagPrediction, defocusVector_test)




    exit(0)


