import numpy as np
import os
import xmippLib as xmipp
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# ---------------------- UTILS METHODS --------------------------------------

def startSessionAndInitialize():
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
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


def make_data_descriptive_plots(df_metadata, folder, COLUMNS, trainDefocus=True, groundTruth=False):
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
    plt.scatter(x, prediction[:, 0], c='b', label='Predicted dU', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Defocus U")
    plt.grid(True)
    plt.legend()
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(x, real[:, 1], c='r', label='Real dV', marker='o')
    plt.scatter(x, prediction[:, 1], c='b', label='Predicted dV', marker='x')
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
    plt.scatter(real[:, 0], prediction[:, 0])
    plt.plot([0, max(real[:, 0])], [0, max(real[:, 0])], color='red', linestyle='--')  # Line for perfect correlation
    plt.xlabel('True Values [defocus U]')
    plt.ylabel('Predictions [defocus U]')
    plt.xlim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(real[:, 1], prediction[:, 1])
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
    error_u = prediction[:, 0] - real[:, 0]
    plt.hist(error_u, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus U")
    plt.ylabel("Count")
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    error_v = prediction[:, 1] - real[:, 1]
    plt.hist(error_v, bins=25, color='green', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus V")
    plt.ylabel("Count")
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'defocus_prediction_error.png'))


def make_testing_angle_plots(prediction, real, folder):
    prediction = np.squeeze(prediction)
    real = np.squeeze(real)

    x = range(1, len(real) + 1)
    # DEFOCUS ANGLE PLOT
    plt.figure(figsize=(16, 8))
    # Plot for angle
    plt.title('Predicted vs real Angle)')
    plt.scatter(x, real, c='r', label='Real angle', marker='o')
    plt.scatter(x, prediction, c='b', label='Predicted angle', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Angle)")
    plt.legend()
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'predicted_vs_real_def_angle.png'))

    # DEFOCUS ANGLE PREDICTED VS REAL
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Angle
    plt.title('Correlation angle')
    plt.scatter(real, prediction)
    plt.xlabel('True Values angle')
    plt.ylabel('Predictions angle')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([0, max(real)], [0, max(real)], color='red', linestyle='--')  # Line for perfect correlation
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'correlation_test_def_angle.png'))

    # DEFOCUS ANGLE ERROR
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Angle
    plt.title('Angle Prediction Error')
    error_sin = prediction - real
    plt.hist(error_sin, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
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
    angleVector = np.zeros((Ndim, 1), dtype=np.float64)
    i = 0

    for index in df.index.to_list():
        storedFile = df.loc[index, 'FILE']
        subset = df.loc[index, 'SUBSET']
        defocus_U = df.loc[index, 'DEFOCUS_U']
        defocus_V = df.loc[index, 'DEFOCUS_V']
        angle = df.loc[index, 'Angle']

        img2 = xmipp.Image(storedFile).getData()

        imagMatrix[i, :, :, 0] = img2

        defocusVector[i, 0] = int(defocus_U)
        defocusVector[i, 1] = int(defocus_V)

        angleVector[i, 0] = angle

        i += 1

    return imagMatrix, defocusVector, angleVector


def centerWindow(image_path, objective_res=2, sampling_rate=1):
    img = xmipp.Image(image_path)
    img_data = img.getData()
    xDim = np.shape(img_data)[1]
    window_size = int(xDim * (sampling_rate / objective_res))
    # Calculate the center coordinates
    center_x, center_y = img_data.shape[0] // 2, img_data.shape[1] // 2
    # Calculate the half-size of the window
    half_window_size = window_size // 2
    # Extract the center window
    window_img = img.window2D(center_x - half_window_size + 1, center_y - half_window_size + 1,
                              center_x + half_window_size, center_y + half_window_size)

    window_data = window_img.getData()

    image_norm = (window_data - np.mean(window_data)) / np.std(window_data)

    # window_image = image_norm[center_x - half_window_size:center_x + half_window_size,
    #                 center_y - half_window_size:center_y + half_window_size]

    return image_norm

def rotation(image_path, angle):
    '''Rotate a np.array and return also the transformation matrix
    #imag: np.array
    #angle: angle in degrees
    #shape: output shape
    #P: transform matrix (further transformation in addition to the rotation)'''
    from scipy.ndimage import rotate

    img = xmipp.Image(image_path)
    image = img.getData()

    rotated_image = rotate(image, angle=angle, reshape=False)

    image_transformed = xmipp.Image()
    image_transformed.setData(rotated_image)

    return image_transformed

def sum_angles(angle1, angle2):
    # Sum the angles
    total_angle = angle1 + angle2
    # Use module to reset the sum to 0 when it reaches or exceeds 180 degrees
    total_angle = total_angle % 180

    return total_angle

def call_ctf_function(kV, sampling_rate, size, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    if kV == 200:
        e_wavelength = 2.75e-2
    else:
        e_wavelength = 2.24e-2

    x = np.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size)
    y = np.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size)

    # Generate x, y values for a grid
    X, Y = np.meshgrid(x, y)

    # Calculate function values for the chosen parameters
    ctf_values = ctf_function(X, Y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast)

    return ctf_values

def ctf_function(x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    angle_g = np.arctan2(y, x)
    angle_ast = np.radians(angle_ast).astype(np.float32)

    dz = (defocusU * (np.cos(angle_g - angle_ast) ** 2) + defocusV * (np.sin(angle_g - angle_ast) ** 2)).astype(np.float32)
    freq = np.sqrt((x ** 2) + (y ** 2)).astype(np.float32)

    # print("NumPy - angle_g:", angle_g, "angle_ast:", angle_ast, "dz:", dz, "freq:", freq)

    return ctf_1d(dz, lambda_e=e_wavelength, freq=freq, cs=Cs)

def ctf_1d(dz, lambda_e, freq, cs):
    term1 = np.pi * dz * lambda_e * (freq**2)
    term2 = np.pi / 2 * cs * (lambda_e**3) * (freq**4)
    # print("NumPy - term1:", term1, "term2:", term2)

    return -np.cos(term1 - term2).astype(np.float32)

def call_ctf_function_tf(kV, sampling_rate, size, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    if kV == 200:
        e_wavelength = 2.75e-2
    else:
        e_wavelength = 2.24e-2

    # Generate x, y values for a grid using TensorFlow
    x = tf.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size)
    y = tf.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size)

    # Generate grid using TensorFlow meshgrid
    X, Y = tf.meshgrid(x, y)

    # Calculate function values for the chosen parameters
    ctf_values = ctf_function_tf(X, Y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast)

    return ctf_values
def ctf_function_tf(x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    angle_g = tf.atan2(y, x)
    # Assuming angle_ast is a tensor
    angle_ast = tf.multiply(angle_ast, np.pi / 180.0)
    dz = defocusU * tf.math.square(tf.math.cos(angle_g - angle_ast)) + defocusV * tf.math.square(tf.math.sin(angle_g - angle_ast))
    freq = tf.sqrt(tf.square(x) + tf.square(y))
    # print("TF - angle_g:", angle_g, "angle_ast:", angle_ast, "dz:", dz, "freq:", freq)

    return ctf_1d_tf(dz, lambda_e=e_wavelength, freq=freq, cs=Cs)

def ctf_1d_tf(dz, lambda_e, freq, cs):
    term1 = tf.multiply(tf.constant(np.pi, dtype=tf.float32), tf.multiply(tf.multiply(dz, lambda_e), tf.square(freq)))
    term2 = tf.multiply(tf.constant(np.pi / 2, dtype=tf.float32), tf.multiply(tf.multiply(cs, tf.pow(lambda_e, 3)), tf.pow(freq, 4)))
    # print("TF - term1:", term1, "term2:", term2)

    return -tf.cos(term1 - term2)

def normalize_angle(angle_degrees):
    # Step 1: Normalize to [0, 1]
    normalized_angle = angle_degrees % 360.0 / 360.0

    # Step 2: Map to [-1, 1]
    normalized_angle_mapped = 2.0 * normalized_angle - 1.0

    return normalized_angle_mapped