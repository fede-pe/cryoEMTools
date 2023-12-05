import numpy as np
import os
import math
import xmippLib as xmipp
import tensorflow as tf
import matplotlib.pyplot as plt


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
    X_set_generated = np.zeros((len(X) * ops, 512, 512, 1))
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

def make_data_descriptive_plots(df_metadata, folder, COLUMNS , trainDefocus = True, trainAngle = True, groundTruth = False):
    if trainDefocus:
        # HISTOGRAM
        df_defocus = df_metadata[[COLUMNS['defocus_U'], COLUMNS['defocus_V']]]
        df_defocus.plot.hist(alpha=0.5, bins=25)
        plt.title('Defocus histogram')
        plt.savefig(os.path.join(folder, 'defocus_histogram.png'))
        # BOXPLOT
        df_defocus.plot.box()
        plt.title('Defocus boxplot')
        plt.savefig(os.path.join(folder, 'defocus_boxplot.png'))
        # SCATTER
        df_defocus.plot.scatter(x=COLUMNS['defocus_U'], y=COLUMNS['defocus_V'])
        plt.title('Correlation plot defocus U vs V')
        plt.plot([0, df_defocus[COLUMNS['defocus_U']].max()],
                 [0, df_defocus[COLUMNS['defocus_U']].max()],
                 color='red')
        plt.savefig(os.path.join(folder, 'defocus_correlation.png'))

        if groundTruth:
            df_defocus['ErrorU'] = df_metadata[COLUMNS['defocus_U']] - df_metadata['DEFOCUS_U_Est']
            df_defocus['ErrorV'] = df_metadata[COLUMNS['defocus_V']] - df_metadata['DEFOCUS_V_Est']

            df_defocus[['ErrorU', 'ErrorV']].plot.hist(alpha=0.5, bins=25)
            plt.title('Defocus error histogram')
            plt.savefig(os.path.join(folder, 'defocus_error_hist.png'))
            # BOXPLOT
            plt.figure()
            df_defocus[['ErrorU', 'ErrorV']].plot.box()
            plt.title('Defocus error boxplot')
            plt.savefig(os.path.join(folder, 'defocus_error_boxplot.png'))

        print(df_defocus.describe())

    if trainAngle:
        # TODO: more Angles plots
        # HISTOGRAM
        df_angle = df_metadata[[COLUMNS['angle'], COLUMNS['cosAngle'], COLUMNS['sinAngle']]]
        # df_angle[COLUMNS['angle']].plot.hist(alpha=0.5, bins=25)
        plt.figure()
        plt.hist(df_angle[COLUMNS['angle']], bins=25, alpha=0.5, color='skyblue', edgecolor='black')
        plt.title('Angle Histogram')
        plt.xlabel('Angle')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(folder, 'angle_histogram.png'))
        print(df_angle.head())

        if groundTruth:
            df_angle_error = df_metadata[COLUMNS['angle']] - df_metadata['Angle_Est']
            plt.figure()
            df_angle_error.plot.hist(alpha=0.5, bins=25)
            plt.title('Angle error')
            plt.savefig(os.path.join(folder, 'angle_error_hist.png'))
            print('Df angle error')
            print(df_angle_error.describe())

def make_training_plots(history, folder, prefix):
    # plot loss during training to CHECK OVERFITTING
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(os.path.join(folder, prefix + 'Training_and_Loss.png'))
    # plt.show()

    # Plot Learning Rate decreasing
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    # Plot Learning Rate
    plt.plot(history.epoch, history.history["lr"], "bo-", label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate", color='b')
    plt.tick_params(axis='y', colors='b')
    plt.grid(True)
    plt.title("Learning Rate and Validation Loss", fontsize=14)
    # Create a twin Axes sharing the xaxis
    ax2 = plt.gca().twinx()
    # Plot Validation Loss
    ax2.plot(history.epoch, history.history["val_loss"], "r^-", label="Validation Loss")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params(axis='y', colors='r')
    # Ensure both legends are displayed
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    # Save the figure
    plt.savefig(os.path.join(folder, prefix + 'Reduce_lr.png'))
    # plt.show()


def make_testing_plots(prediction, real, folder):
    # DEFOCUS PLOT
    plt.figure(figsize=(16, 8))  # Adjust the figure size as needed
    # Plot for Defocus U
    plt.subplot(211)
    plt.title('Defocus U')
    x = range(1, len(real[:, 0]) + 1)
    plt.scatter(x, real[:, 0], c='r', label='Real dU', marker='o')
    plt.scatter(x, prediction[0], c='b', label='Predicted dU', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Defocus U")
    plt.grid(True)
    plt.legend()
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(x, real[:, 1], c='r', label='Real dV', marker='o')
    plt.scatter(x, prediction[1], c='b', label='Predicted dV', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Defocus V")
    plt.grid(True)
    plt.legend()
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'predicted_vs_real_def.png'))

    # CORRELATION PLOT
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Defocus U
    plt.subplot(211)
    plt.title('Defocus U')
    plt.scatter(real[:, 0], prediction[0])
    plt.plot([0, max(real[:, 0])], [0, max(real[:, 0])], color='red', linestyle='--')  # Line for perfect correlation
    plt.xlabel('True Values [defocus U]')
    plt.ylabel('Predictions [defocus U]')
    plt.xlim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(real[:, 1], prediction[1])
    plt.plot([0, max(real[:, 0])], [0, max(real[:, 0])], color='red', linestyle='--')  # Line for perfect correlation
    plt.xlabel('True Values [defocus V]')
    plt.ylabel('Predictions [defocus V]')
    plt.xlim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'correlation_test_def.png'))
    # plt.show()

    # DEFOCUS ERROR
    # Plot for Defocus U
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('Defocus U')
    error_u = prediction[0] - real[:, 0].reshape(-1, 1)
    plt.hist(error_u, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus U")
    plt.ylabel("Count")
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    error_v = prediction[1] - real[:, 1].reshape(-1, 1)
    plt.hist(error_v, bins=25, color='green', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus V")
    plt.ylabel("Count")
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'defocus_prediction_error.png'))


def make_testing_angle_plots(prediction, real, folder):
    x = range(1, len(real[:, 0]) + 1)
    # DEFOCUS ANGLE PLOT
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Sin(2*angle)
    plt.subplot(211)
    plt.title('Sin(2*angle)')
    plt.scatter(x, real[:, 0], c='r', label='Real Sin', marker='o')
    plt.scatter(x, prediction[0], c='b', label='Predicted Sin', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Sin(2*angle)")
    plt.legend()
    # Plot for Cos(2*angle)
    plt.subplot(212)
    plt.title('Cos(2*angle)')
    plt.scatter(x, real[:, 1], c='r', label='Real Cos', marker='o')
    plt.scatter(x, prediction[1], c='b', label='Predicted Cos', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Cos(2*angle)")
    plt.legend()
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'predicted_vs_real_def.png'))

    # DEFOCUS ANGLE PREDICTED VS REAL !OJO NO VA MUY BIEN ESTE PLOT
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Sin(2*angle)
    plt.subplot(211)
    plt.title('Sin(2 * angle)')
    plt.scatter(real[:, 0], prediction[0])
    plt.xlabel('True Values [Sin]')
    plt.ylabel('Predictions [Sin]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100], color='red', linestyle='--')  # Line for perfect correlation
    # Plot for Cos(2*angle)
    plt.subplot(212)
    plt.title('Cos(2 * angle)')
    plt.scatter(real[:, 1], prediction[1])
    plt.xlabel('True Values [Cos]')
    plt.ylabel('Predictions [Cos]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100], color='red', linestyle='--')  # Line for perfect correlation
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'correlation_test_def_angle.png'))

    # DEFOCUS ANGLE ERROR
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Sin(2*Angle)
    plt.subplot(211)
    plt.title('Sin(2*Angle) Prediction Error')
    error_sin = prediction[0] - real[:, 0].reshape(-1, 1)
    plt.hist(error_sin, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    # Plot for Cos(2*Angle)
    plt.subplot(212)
    plt.title('Cos(2*Angle) Prediction Error')
    error_cos = prediction[1] - real[:, 1].reshape(-1, 1)
    plt.hist(error_cos, bins=25, color='orange', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'defocus_angle_prediction_error.png'))

def prepareTestData(df):
    Ndim = df.shape[0]
    imagMatrix = np.zeros((Ndim, 512, 512, 1), dtype=np.float64)
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
        # img1Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_1.xmp")
        img2Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_2.xmp")
        # img3Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_3.xmp")

        # img1 = xmipp.Image(img1Path).getData()
        img2 = xmipp.Image(img2Path).getData()
        # img3 = xmipp.Image(img3Path).getData()

        # imagMatrix[i, :, :, 0] = img1
        imagMatrix[i, :, :, 0] = img2
        # imagMatrix[i, :, :, 2] = img3

        defocusVector[i, 0] = int(defocus_U)
        defocusVector[i, 1] = int(defocus_V)

        angleVector[i, 0] = sinAngle
        angleVector[i, 1] = cosAngle

        i += 1

    return imagMatrix, defocusVector, angleVector