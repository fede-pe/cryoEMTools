import numpy as np
import os
import math
import xmippLib as xmipp
import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:XLA_GPU:0"
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from DeepDefocusModel import DeepDefocusMultiOutputModel

BATCH_SIZE = 8  # Tiene que ser multiplos de tu tamaño de muestra
EPOCHS = 100
LEARNING_RATE = 0.001
IM_WIDTH = 512
IM_HEIGHT = 512
training_Bool = True
testing_Bool = True
plots_Bool = True
TEST_SIZE = 0.15


# ---------------------- UTILS METHODS --------------------------------------

def startSessionAndInitialize():
    print('Enable dynamic memory allocation')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


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
    ops = 1  # number of operations per image
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


def make_testing_plots(prediction, real):
    # DEFOCUS PLOT
    x = range(1, len(real[:, 0]) + 1)
    plt.subplot(211)
    plt.title('Defocus U')
    plt.scatter(x, real[:, 0], c='r', label='dU')
    plt.scatter(x, prediction[0], c='b', label='dU_pred')
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(x, real[:, 1], c='r', label='dV')
    plt.scatter(x, prediction[1], c='b', label='dV_pred')
    plt.legend()
    plt.show()

    # DEFOCUS PREDICTED VS REAL
    plt.subplot(211)
    plt.title('Defocus U')
    plt.scatter(real[:, 0], prediction[0])
    plt.xlabel('True Values [defocus U]')
    plt.ylabel('Predictions [defocus U]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(real[:, 1], prediction[1])
    plt.xlabel('True Values [defocus V]')
    plt.ylabel('Predictions [defocus v]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    # DEFOCUS ERROR
    plt.subplot(211)
    plt.title('Defocus U')
    error = prediction[0] - real[:, 0]
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error Defocus U")
    _ = plt.ylabel("Count")
    plt.subplot(212)
    plt.title('Defocus V')
    error = prediction[1] - real[:, 1]
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error Defocus V")
    plt.show()


def make_testing_angle_plots(prediction, real):
    x = range(1, len(real[:, 0]) + 1)
    # DEFOCUS ANGLE PLOT
    plt.subplot(211)
    plt.title('Sin(2*angle)')
    plt.scatter(x, real[:, 0], c='r', label='Sin')
    plt.scatter(x, prediction[:, 0], c='b', label='Sin_pred')
    plt.subplot(212)
    plt.title('Cos(2*angle)')
    plt.scatter(x, real[:, 1], c='r', label='Cos')
    plt.scatter(x, prediction[:, 1], c='b', label='Cos_pred')
    plt.legend()
    plt.show()


    # DEFOCUS ANGLE PREDICTED VS REAL !OJO NO VA MUY BIEN ESTE PLOT
    plt.subplot(211)
    plt.title('Sin ( 2 * angle)')
    plt.scatter(real[:, 0], prediction[:, 0])
    plt.xlabel('True Values [Sin]')
    plt.ylabel('Predictions [Sin]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.subplot(212)
    plt.title('Cos (2 * angle)')
    plt.scatter(real[:, 1], prediction[:, 1])
    plt.xlabel('True Values [Cos]')
    plt.ylabel('Predictions [Cos]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


    # DEFOCUS ANGLE ERROR
    plt.subplot(211)
    plt.title('Sin(2*Angle)')
    error = prediction[:, 0] - real[:, 0]
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.subplot(212)
    plt.title('Cos(2*Angle)')
    error = prediction[:, 1] - real[:, 1]
    plt.hist(error, bins=0.2)
    plt.xlabel("Prediction Error")
    plt.show()


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


def getModelSeparatedDefocus():
    model = DeepDefocusMultiOutputModel().assemble_model_separated_defocus(IM_WIDTH, IM_HEIGHT)
    model.summary()
    # plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss={'defocus_U_output': 'mae', 'defocus_V_output': 'mae'},
                  loss_weights=None,
                  metrics={})

    return model


def getModelDefocusAngle():
    model = DeepDefocusMultiOutputModel().assemble_model_angle(IM_HEIGHT, IM_WIDTH)
    model.summary()
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss={'defocus_angle_output': 'mae'},
                  loss_weights=None,
                  metrics={})  # Añadir una metrica q sea en funcion del angulo despejado

    return model


def prepareTestData(df):
    Ndim = df.shape[0]
    imagMatrix = np.zeros((Ndim, 512, 512, 3), dtype=np.float64)
    defocusVector = np.zeros((Ndim, 2), dtype=np.float64)
    angleVector = np.zeros((Ndim, 2), dtype=np.float64)
    i = 0

    for index in df.index.to_list():
        storedFile = df.loc[index, 'FILE']
        subset = df.loc[index, 'SUBSET']
        defocus_U = df.loc[index, 'DEFOCUS_U']
        defocus_V = df.loc[index, 'DEFOCUS_V']
        sinAngle = df.loc[index, 'Sin(2*angle)']
        cosAngle = df.loc[index, 'Cos(2*angle)']
        # Replace is done since we want the 3 images not only the one in the metadata file
        img1Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_1.xmp")
        img2Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_2.xmp")
        img3Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_3.xmp")

        img1 = xmipp.Image(img1Path).getData()
        img2 = xmipp.Image(img2Path).getData()
        img3 = xmipp.Image(img3Path).getData()

        imagMatrix[i, :, :, 0] = img1
        imagMatrix[i, :, :, 1] = img2
        imagMatrix[i, :, :, 2] = img3

        defocusVector[i, 0] = int(defocus_U)
        defocusVector[i, 1] = int(defocus_V)

        angleVector[i, 0] = sinAngle
        angleVector[i, 1] = cosAngle

        i += 1

    return imagMatrix, defocusVector, angleVector