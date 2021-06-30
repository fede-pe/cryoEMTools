#!/usr/bin/env python2
import numpy as np
import os
import sys
from time import time
import math
import xmippLib as xmipp
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:XLA_GPU:0"
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from createDeepDefocusModel import DeepDefocusMultiOutputModel
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
IM_WIDTH = 512
IM_HEIGHT = 512
training_Bool = False
testing_Bool = False
plots_Bool = True
TEST_SIZE = 0.15


# ---------------------- UTILS METHODS --------------------------------------
def applyTransform(imag_array, M, shape):
    '''Apply a transformation(M) to a np array(imag) and return it in a given shape'''
    imag = xmipp.Image()
    imag.setData(imag_array)
    imag = imag.applyWarpAffine(list(M.flatten()), shape, True)
    return imag.getData()


def rotation(imag, angle, shape, P):
    '''Rotate a np.array and return also the transformation matrix
    #imag: np.array
    #angle: angle in degrees
    #shape: output shape
    #P: transform matrix (further transformation in addition to the rotation)'''
    (hsrc, wsrc) = imag.shape
    angle *= math.pi / 180
    T = np.asarray([[1, 0, -wsrc / 2], [0, 1, -hsrc / 2], [0, 0, 1]])
    R = np.asarray([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    M = np.matmul(np.matmul(np.linalg.inv(T), np.matmul(R, T)), P)

    transformed = applyTransform(imag, M, shape)
    return transformed, M


def data_generator(X, Y, rotation_angle=90):
    ops = 2 #number of operations per image
    X_set_generated = np.zeros((len(X) * ops, IM_HEIGHT, IM_WIDTH, 3))
    Y_set_generated = np.zeros((len(Y) * ops, 2))
    P = np.identity(3)

    for i, j in zip(range(len(X) - 1), range(0, len(X_set_generated) - 1, 2)):
        for n in range(ops):
            X_set_generated[j + n, :, :, 0], _ = rotation(X[i, :, :, 0], rotation_angle * n,
                                                          X[i, :, :, 0].shape, P)
            X_set_generated[j + n, :, :, 1], _ = rotation(X[i, :, :, 1], rotation_angle * n,
                                                          X[i, :, :, 1].shape, P)
            X_set_generated[j + n, :, :, 2], _ = rotation(X[i, :, :, 2], rotation_angle * n,
                                                          X[i, :, :, 2].shape, P)
            Y_set_generated[j + n, :] = Y[i, :]

    return X_set_generated, Y_set_generated


def make_training_plots(history):
    # plot loss during training to CHECK OVERFITTING
    plt.title('Loss')
    plt.plot(history.history['loss'], 'b', label='training loss')
    plt.plot(history.history['val_loss'], 'r', label='validation loss')
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
    x = range(1, len(defocusVector[:, 0]) + 1)
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
    # plt.subplot(211)
    # plt.title('Sin(2*angle)')
    # plt.scatter(x, defocusVector[:, 2], c='r', label='Sin')
    # plt.scatter(x, imagPrediction[:, 2], c='b', label='Sin_pred')
    # plt.subplot(212)
    # plt.title('Cos(2*angle)')
    # plt.scatter(x, defocusVector[:, 3], c='r', label='Cos')
    # plt.scatter(x, imagPrediction[:, 3], c='b', label='Cos_pred')
    # plt.legend()
    # plt.show()

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
    # plt.subplot(211)
    # plt.title('Sin ( 2 * angle)')
    # plt.scatter(defocusVector[:, 2], imagPrediction[:, 2])
    # plt.xlabel('True Values [Sin]')
    # plt.ylabel('Predictions [Sin]')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlim([0, plt.xlim()[1]])
    # plt.ylim([0, plt.ylim()[1]])
    # _ = plt.plot([-100, 100], [-100, 100])
    # plt.subplot(212)
    # plt.title('Cos (2 * angle)')
    # plt.scatter(defocusVector[:, 3], imagPrediction[:, 3])
    # plt.xlabel('True Values [Cos]')
    # plt.ylabel('Predictions [Cos]')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlim([0, plt.xlim()[1]])
    # plt.ylim([0, plt.ylim()[1]])
    # _ = plt.plot([-100, 100], [-100, 100])
    # plt.show()

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
    # plt.subplot(211)
    # plt.title('Sin(2*Angle)')
    # error = imagPrediction[:, 2] - defocusVector[:, 2]
    # plt.hist(error, bins=25)
    # plt.xlabel("Prediction Error")
    # _ = plt.ylabel("Count")
    # plt.subplot(212)
    # plt.title('Cos(2*Angle)')
    # error = imagPrediction[:, 3] - defocusVector[:, 3]
    # plt.hist(error, bins=25)
    # plt.xlabel("Prediction Error")
    # plt.show()


# ----------- MODEL ARCHITECTURE  -------------------
def getModel():
    model = DeepDefocusMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)
    model.summary()
    # plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss={'defocus_output': 'mae',
                        'defocus_angles_output': 'mae'},
                  loss_weights=None,
                  metrics={})  # 'defocus_output': 'msle',
    # 'defocus_angles_output': 'msle'})

    return model


def getModelDefocus():
    model = DeepDefocusMultiOutputModel().assemble_model_defocus(IM_WIDTH, IM_HEIGHT)
    model.summary()
    # plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss={'defocus_output': 'mae'},
                  loss_weights=None,
                  metrics={})

    return model


# ------------------------ MAIN PROGRAM -----------------------------
if __name__ == "__main__":

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

    # ----------- PREPROCESSING ------------------
    # Normalized data
    std_im = imagMatrix.std()
    mean_im = imagMatrix.mean()
    imagMatrix_Norm = (imagMatrix - mean_im) / std_im

    mean_def_U = defocusVector[:, 0].mean()
    std_def_U = defocusVector[:, 0].std()
    min_def_U = defocusVector[:, 0].min()
    max_def_U = defocusVector[:, 0].max()

    mean_def_V = defocusVector[:, 1].mean()
    std_def_V = defocusVector[:, 1].std()
    min_def_V = defocusVector[:, 1].min()
    max_def_V = defocusVector[:, 1].max()

    corr = np.corrcoef(defocusVector[:, 0], defocusVector[:, 1])

    defocusVector_Norm = np.zeros((len(defocusVector), 2))

    # ---------------- STATISTICS ------------------------
    print('STATISTICS')
    print('Correlation: ' + str(corr))
    print('Defocus U')
    print('Mean: ' + str(mean_def_U))
    print('Std: ' + str(std_def_U))
    print('Min: ' + str(min_def_U))
    print('Max: ' + str(max_def_U))
    print('Defocus V')
    print('Mean: ' + str(mean_def_V))
    print('Std: ' + str(std_def_V))
    print('Min: ' + str(min_def_V))
    print('Max: ' + str(max_def_V))

    # Histogram plot
    plt.style.use('ggplot')
    plt.subplot(211)
    plt.title('Defocus U')
    plt.hist(defocusVector[:, 0], bins=50, label='dU', color='b')
    plt.subplot(212)
    plt.title('Defocus V')
    plt.hist(defocusVector[:, 1], bins=50, label='dV', color='c')
    plt.show()

    # Correlation plot
    plt.title('Pearson correlation = ' + "{:.2f}".format(corr[0, 1]))
    plt.scatter(defocusVector[:, 0], defocusVector[:, 1])
    plt.xlabel('[defocus U]')
    plt.ylabel('[defocus V]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    # ------------------ NORMALIZING DATA -----------------------------------
    defocusVector_Norm[:, 0] = (defocusVector[:, 0] - mean_def_U) / std_def_U
    defocusVector_Norm[:, 1] = (defocusVector[:, 1] - mean_def_V) / std_def_V

    print('Input matrix: ' + str(np.shape(imagMatrix_Norm)))
    print('Output matrix: ' + str(np.shape(defocusVector_Norm)))

    # DATA GENERATOR
    print('Generating images...')
    X_set_generated, Y_set_generated = data_generator(imagMatrix_Norm, defocusVector_Norm)
    print('Input generated matrix: ' + str(np.shape(X_set_generated)))
    print('Output generated matrix: ' + str(np.shape(Y_set_generated)))

    # For applying two branches one for the defocus and the other for the angle
    # defocusVector_train_tmp = [defocusVector_train[:, :2], defocusVector_train[:, 2:]]
    # print('Output train tmp matrix: ' + str(np.shape(defocusVector_train_tmp)))

    # SPLIT INTO TRAIN AND TEST
    n = len(X_set_generated)
    print('Split data into train and test')
    # imagMatrix_train, imagMatrix_test = X_set_generated[:int(n*(1-TEST_SIZE)), :, :, :], X_set_generated[int(n*(1-TEST_SIZE)):, :, :, :]
    # defocusVector_train, defocusVector_test = Y_set_generated[:int(n*(1-TEST_SIZE)), :], Y_set_generated[int(n*(1-TEST_SIZE)):, :]

    # Funciona solo con cierta cantidad de datos luego te quedas sin memoria RAM
    imagMatrix_train, imagMatrix_test, defocusVector_train, defocusVector_test = \
        train_test_split(X_set_generated, Y_set_generated, test_size=0.15, random_state=42)

    print('Input train matrix: ' + str(np.shape(imagMatrix_train)))
    print('Output train matrix: ' + str(np.shape(defocusVector_train)))
    print('Input test matrix: ' + str(np.shape(imagMatrix_test)))
    print('Output test matrix: ' + str(np.shape(defocusVector_test)))

    # ----------- TRAINING MODEL-------------------
    if training_Bool:
        print("Train mode")
        start_time = time()
        model = getModelDefocus()

        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

        callbacks_list = [callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1'), separator=',', append=False),
                          callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1'), histogram_freq=0,
                                                batch_size=BATCH_SIZE,
                                                write_graph=True, write_grads=False, write_images=False,
                                                embeddings_freq=0, embeddings_layer_names=None,
                                                embeddings_metadata=None, embeddings_data=None),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0),
                          callbacks.EarlyStopping(monitor='val_loss', patience=10)
                          ]

        history = model.fit(imagMatrix_train, defocusVector_train,
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
        np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPrediction)

        # DEFOCUS
        # mae = mean_absolute_error(defocusVector_test[:, :2], imagPredictionTmp[:, :2])
        # print("Defocus model mean absolute error val_loss: ", mae)

        # DEFOCUS_ANGLE
        # mae = mean_absolute_error(defocusVector_test[:, 2:], imagPredictionTmp[:, 2:])
        # print("Defocus angle model mean absolute error val_loss: ", mae)

        # TOTAL ERROR
        defocusVector_test_rescaled = np.zeros((len(defocusVector_test), 2))
        imagPrediction_rescaled = np.zeros((len(imagPrediction), 2))

        defocusVector_test_rescaled[:, 0] = (defocusVector_test[:, 0] * std_def_U) + mean_def_U
        defocusVector_test_rescaled[:, 1] = (defocusVector_test[:, 1] * std_def_V) + mean_def_V
        imagPrediction_rescaled[:, 0] = (imagPrediction[:, 0] * std_def_U) + mean_def_U
        imagPrediction_rescaled[:, 1] = (imagPrediction[:, 1] * std_def_V) + mean_def_V

        mae = mean_absolute_error(defocusVector_test_rescaled, imagPrediction_rescaled)
        print("Final model mean absolute error val_loss: ", mae)

        # MSE
        loss = model.evaluate(imagMatrix_test, defocusVector_test, verbose=2)
        print("Testing set Total Mean Abs Error: {:5.2f} charges".format(loss))
        # print("Testing set Defocus Mean Abs Error: {:5.2f} charges".format(loss_defocus))
        # print("Testing set Angle Mean Abs Error: {:5.2f} charges".format(loss_angle))

        if plots_Bool:
            make_testing_plots(imagPrediction_rescaled, defocusVector_test_rescaled)

    exit(0)
