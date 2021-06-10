#!/usr/bin/env python2
import numpy as np
import os
import sys
from time import time
# ----TENSORFLOW INSIDE KERAS
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:XLA_GPU:0"
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from createDeepDefocusModel import DeepDefocusMultiOutputModel

from tensorflow.keras.utils import plot_model



BATCH_SIZE = 128  # 128 should be by default (The higher the faster it converge)
EPOCHS = 100
LEARNING_RATE = 0.0004
IM_WIDTH = 512
IM_HEIGHT = 512
training_Bool = False
testing_Bool = True
plots_Bool = True
TEST_SIZE = 0.15


# ------------------------ MAIN PROGRAM -----------------------------

if __name__ == "__main__":

    # ---------------------- UTILS METHODS --------------------------------------

    def make_training_plots(history):
        # plot loss during training to CHECK OVERFITTING
        plt.title('Loss')
        plt.plot(history.history['loss'], 'b+', label='training loss')
        plt.plot(history.history['val_loss'], 'r+', label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
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
        L = MaxPooling2D()(L) #falta el la capa DENSE para el dropout
        L = Dropout(0.2)(L)
        L = Flatten()(L) #Here is where we need to put more dense layers
        L = Dense(4, name="output", activation="linear")(L)

        model = Model(inputLayer, L)
        model.summary()
        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae', 'msle'])  #MAE is more robust to outliers

        return model


    def getModel():
        model = DeepDefocusMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)
        model.summary()
        plot_model(model, to_file='test.png', show_shapes=True)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer,
                      loss={
                          'defocus_U_output': 'mae',
                          'defocus_V_output': 'mae',
                          'defocus_Cosangles_output': 'msle',
                          'defocus_Sinangles_output': 'msle'},
                      loss_weights=None, # {
                          #'defocus_U_output': 0.25,
                         # 'defocus_V_output': 0.25,
                         # 'defocus_Cosangles_output': 0.25,
                         # 'defocus_Sinangles_output': 0.25},
                      metrics={})

        return model


    def getModel2():
        model = DeepDefocusMultiOutputModel().assemble_full_model_original(IM_WIDTH, IM_HEIGHT)
        model.summary()
        #plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer,
                      loss={'defocus_output': 'mae',
                            'defocus_angles_output': 'mae'},
                      loss_weights=None,
                      metrics={'defocus_output': 'msle',
                            'defocus_angles_output': 'msle'})

        return model


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

    print('Input train matrix: ' + str(np.shape(imagMatrix_train)))
    print('Input test matrix: ' + str(np.shape(imagMatrix_test)))
    print('Output train matrix: ' + str(np.shape(defocusVector_train)))
    print('Output test matrix: ' + str(np.shape(defocusVector_test)))



    #defocusVector_train_tmp = np.array([defocusVector_train[:,:2], defocusVector_train[:,2:]])
    defocusVector_train_tmp = [defocusVector_train[:, :2], defocusVector_train[:, 2:]]
    print('Output train tmp matrix: ' + str(np.shape(defocusVector_train_tmp)))
    print(defocusVector_train_tmp)

# ----------- TRAINING MODEL-------------------
    if training_Bool:
        print("Train mode")
        start_time = time()
        model = getModel2()

        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

        callbacks_list = [callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1'), separator=',', append=False),
                          callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1'), histogram_freq=0,
                                                write_graph=True, write_grads=False, write_images=False,
                                                embeddings_freq=0, embeddings_layer_names=None,
                                                embeddings_metadata=None, embeddings_data=None),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0),
                          callbacks.EarlyStopping(monitor='val_loss', patience=10)
                          ]

        history = model.fit(imagMatrix_train, defocusVector_train_tmp, steps_per_epoch=len(imagMatrix_train)//BATCH_SIZE,
                            batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                            validation_split=0.15, callbacks=callbacks_list)

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
        imagPredictionTmp = np.zeros(shape=(np.shape(imagPrediction)[1], 4))
        imagPredictionTmp[:, :2] = imagPrediction[0]
        imagPredictionTmp[:, 2:] = imagPrediction[1]

        np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPredictionTmp)

        # DEFOCUS
        mae = mean_absolute_error(defocusVector_test[:, :2], imagPredictionTmp[:, :2])
        print("Defocus model mean absolute error val_loss: ", mae)

        # DEFOCUS_ANGLE
        mae = mean_absolute_error(defocusVector_test[:, 2:], imagPredictionTmp[:, 2:])
        print("Defocus angle model mean absolute error val_loss: ", mae)

        # TOTAL ERROR
        mae = mean_absolute_error(defocusVector_test, imagPredictionTmp)
        print("Final model mean absolute error val_loss: ", mae)

        print('Test in a different approach')
        loss, loss_defocus, loss_angle = model.evaluate(imagMatrix_test, [defocusVector_test[:, :2],
                                                                          defocusVector_test[:, 2:]], verbose=2)

        print("Testing set Total Mean Abs Error: {:5.2f} charges".format(loss))
        print("Testing set Defocus Mean Abs Error: {:5.2f} charges".format(loss_defocus))
        print("Testing set Angle Mean Abs Error: {:5.2f} charges".format(loss_angle))

        if plots_Bool:
            make_testing_plots(imagPredictionTmp, defocusVector_test)


    exit(0)


