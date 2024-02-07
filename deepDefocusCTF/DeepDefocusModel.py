from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Lambda, Concatenate, Reshape, UpSampling2D, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import centerWindow, call_ctf_function_tf, call_ctf_function
import os
from scipy.stats import pearsonr


def angle_error_metric(y_true, y_pred):
    # Extract angles predicted and true values from the model's output
    angle_true = y_true[:, 2] * 180
    angle_pred = y_pred[:, 2] * 180
    # Calculate the absolute error in degrees
    angle_error_degrees = K.abs(angle_pred - angle_true)
    # Return the mean angle error
    return K.mean(angle_error_degrees)

def mae_defocus_error(y_true, y_pred, defocus_scaler):
    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

    y_true_unscaled = (y_true[:, 0:2] * iqr_) + median_
    y_pred_unscaled = (y_pred[:, 0:2] * iqr_) + median_

    metric_value = tf.reduce_mean(tf.abs(y_true_unscaled - y_pred_unscaled))

    return metric_value

def corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 1
    size = 512
    epsilon = 1e-8

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    angle_true = y_true[:, 2]
    angle_pred = y_pred[:, 2]

    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    # Example: Access individual output tensors
    defocus_U_true = y_true_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true = y_true_unscaled[:, 1]  # Assuming defocus_V is the second output
    defocus_U_pred = y_pred_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred = y_pred_unscaled[:, 1]  # Assuming defocus_V is the second output

    # ------
    def elementwise_loss(defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                         angle_true, angle_pred):
        # Extract true sin and cos values
        # Calculate the true angle
        true_angle = angle_true * 180
        pred_angle = angle_pred * 180

        ctf_array_true = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_true,
                                              defocusV=defocus_V_true, Cs=cs, phase_shift_PP=0, angle_ast=true_angle)

        ctf_array_pred = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_pred,
                                              defocusV=defocus_V_pred, Cs=cs, phase_shift_PP=0, angle_ast=pred_angle)

        # Flatten the arrays to make them 1D
        ctf_array_true_flat = tf.reshape(ctf_array_true, [-1])
        ctf_array_pred_flat = tf.reshape(ctf_array_pred, [-1])

        # Calculate mean-centered vectors
        mean_true = tf.reduce_mean(ctf_array_true_flat)
        mean_pred = tf.reduce_mean(ctf_array_pred_flat)

        centered_true = ctf_array_true_flat - mean_true
        centered_pred = ctf_array_pred_flat - mean_pred

        # Calculate Pearson correlation coefficient
        numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))

        correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)
        correlation_coefficient_loss = 1 - correlation_coefficient

        return correlation_coefficient_loss

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    return tf.reduce_mean(elementwise_losses)

class DeepDefocusMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one for defocus
    and another for the defocus angles. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """

    def __init__(self, width=512, height=512):
        self.IM_WIDTH = width
        self.IM_HEIGHT = height

    def build_defocus_branch(self, input, Xdim, factor):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = (Conv2D(32, (int(Xdim * factor), int(Xdim * factor)), activation="relu", padding="valid", name="conv2d_1")(input))
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((3, 3), name='pool_1')(L)

        Xconv1dim = np.shape(L)[1]

        L = Conv2D(16, (int(Xconv1dim / 10), int(Xconv1dim / 10)), activation="relu", padding="valid", name="conv2d_2")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D(name='pool_2')(L)

        Xconv2dim = np.shape(L)[1]

        L = Conv2D(8, (int(Xconv2dim / 10), int(Xconv2dim / 10)), activation="relu", padding="valid", name="conv2d_3")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D(name='pool_3')(L)

        Xconv3dim = np.shape(L)[1]

        # L = Conv2D(4, (int(Xconv3dim/10), int(Xconv3dim/10)), activation="relu", padding="valid")(L)
        L = Conv2D(4, (3, 3), activation="relu", padding="valid", name="conv2d_4")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D(name='pool_4')(L)

        L = Flatten()(L)

        return L

    def build_defocus_branch_new(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(4, 4),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(4, 4),
                   activation='relu', padding='same', name='conv2d_6'+suffix)(h)
        h = BatchNormalization()(h)
        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_7' + suffix)(h)
        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_8' + suffix)(h)
        h = BatchNormalization()(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h

    def build_defocus_angle_branch(self, input, Xdim, factor):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = Conv2D(16, (int(Xdim * factor), int(Xdim * factor)), activation="relu", padding="valid", name='conv2d_1_ang')(input)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((2, 2))(L)

        Xconv1dim = np.shape(L)[1]

        L = Conv2D(8, (int(Xconv1dim / 10), int(Xconv1dim / 10)), activation="relu", padding="valid", name='conv2d_2_ang')(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        Xconv2dim = np.shape(L)[1]

        L = Conv2D(4, (int(Xconv2dim / 5), int(Xconv2dim / 5)), activation="relu", padding="valid", name='conv2d_3_ang')(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        L = Flatten(name='flatten_ang')(L)

        return L

    def build_angle_branch_new(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
         #Defining the architecture of the CNN
        h = Conv2D(filters=16, kernel_size=(64, 64),
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        #h = Conv2D(filters=16, kernel_size=(50, 50),
        #           activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(4, 4),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        #h = Conv2D(filters=16, kernel_size=(4, 4),
        #           activation='relu', padding='same', name='conv2d_6'+suffix)(h)
        h = BatchNormalization()(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h

    def build_defocusU_branch(self, convLayer):
        L = Dense(64, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(0.01),
                  name="denseU_1")(convLayer)
        L = Dropout(0.1, name="DropU_1")(L)
        L = Dense(16, activation='relu', name="denseU_2")(L)
        # L = Dropout(0.1)(L)
        # L = Dense(4, activation='relu', name="denseU_3")(L)
        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)
        return defocusU

    def build_defocusV_branch(self, convLayer):
        L = Dense(64, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(0.01),
                  name="denseV_1")(convLayer)
        L = Dropout(0.1, name="DropV_1")(L)
        L = Dense(16, activation='relu', name="denseV_2")(L)
        # L = Dropout(0.1)(L)
        # L = Dense(4, activation='relu', name="denseV_3")(L)
        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)
        return defocusV

    def build_angle_branch(self, convLayer):
        L = Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01), name="denseAngle_1")(convLayer)
        L = Dropout(0.1, name="DropA_1")(L)
        L = Dense(16, activation='relu', name="denseAngle_2")(L)

        angle = Dense(1, activation='sigmoid', name='angle_output')(L)

        return angle

    def assemble_model_separated_defocus(self, width, height):
        input_shape = (height, width, 1)
        input_layer = Input(shape=input_shape, name='input')

        # DEFOCUS U and V
        defocus_branch_at_2 = self.build_defocus_branch_new(input_layer, suffix="defocus")
        L = Dropout(0.2)(defocus_branch_at_2)

        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)

        # DEFOCUS ANGLE
        defocus_angles_branch = self.build_angle_branch_new(input_layer, suffix="angle")
        La = Dropout(0.2)(defocus_angles_branch)

        angle = self.build_angle_branch(La)

        # OUTPUT
        concatenated = Concatenate(name='ctf_values')([defocusU, defocusV, angle])

        model = Model(inputs=input_layer, outputs=concatenated,
                      name="deep_separated_defocus_net")

        return model

    # ----------- GET MODEL -------------------
    def getFullModel(self, learning_rate, defocus_scaler, cs, kV):
        model = self.assemble_model_separated_defocus(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

        loss = lambda y_true, y_pred: custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV)

        def angle_error(y_true, y_pred):
            return angle_error_metric(y_true, y_pred)

        def mae_defocus(y_true, y_pred):
            return mae_defocus_error(y_true, y_pred, defocus_scaler)

        def corr_CTF(y_true, y_pred):
            return corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV)

        model.compile(optimizer=optimizer, loss=loss, metrics=[angle_error, mae_defocus, corr_CTF], loss_weights=None)

        return model


# Custom loss function for CTF based on a mathematical formula
def custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 1
    size = 512
    epsilon = 1e-8
    #print("Shape of y_true:", y_true.shape)
    #print("Shape of y_pred:", y_pred.shape)

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    defocus_U_true_scaled = y_true[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true_scaled = y_true[:, 1]  # Assuming defocus_V is the second output

    defocus_U_pred_scaled = y_pred[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred_scaled = y_pred[:, 1]  # Assuming defocus_V is the second output

    angle_true = y_true[:, 2]
    angle_pred = y_pred[:, 2]

    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    #y_true_unscaled = y_true * defocus_scaler
    #y_pred_unscaled = y_pred * defocus_scaler

    # Example: Access individual output tensors
    defocus_U_true = y_true_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true = y_true_unscaled[:, 1]  # Assuming defocus_V is the second output
    defocus_U_pred = y_pred_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred = y_pred_unscaled[:, 1]  # Assuming defocus_V is the second output

    # ------
    def elementwise_loss(defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                         angle_true, angle_pred):
        # Extract true sin and cos values
        # Calculate the true angle
        true_angle = angle_true * 180
        pred_angle = angle_pred * 180

        ctf_array_true = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_true,
                                              defocusV=defocus_V_true, Cs=cs, phase_shift_PP=0, angle_ast=true_angle)

        ctf_array_pred = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_pred,
                                              defocusV=defocus_V_pred, Cs=cs, phase_shift_PP=0, angle_ast=pred_angle)

        # Print intermediate values for debugging
        #tf.print("ctf_array_true:", ctf_array_true)
        #tf.print("ctf_array_pred:", ctf_array_pred)

        # Flatten the arrays to make them 1D
        ctf_array_true_flat = tf.reshape(ctf_array_true, [-1])
        ctf_array_pred_flat = tf.reshape(ctf_array_pred, [-1])

        # Calculate mean-centered vectors
        mean_true = tf.reduce_mean(ctf_array_true_flat)
        mean_pred = tf.reduce_mean(ctf_array_pred_flat)

        centered_true = ctf_array_true_flat - mean_true
        centered_pred = ctf_array_pred_flat - mean_pred

        # Calculate Pearson correlation coefficient
        numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))

        correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)
        correlation_coefficient_loss = 1 - correlation_coefficient

        #return correlation_coefficient_loss
        return tf.abs(ctf_array_true - ctf_array_pred) # MSE or MAE

    #tf.print("defocus_U_true:", defocus_U_true)
    #tf.print("defocus_V_true:", defocus_V_true)
    #tf.print("defocus_U_pred:", defocus_U_pred)
    #tf.print("defocus_V_pred:", defocus_V_pred)
    #tf.print("Angle_true:", angle_true)
    #tf.print("Angle_pred:", angle_pred)


    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    defocus_U_loss = tf.reduce_mean(tf.abs(defocus_U_true_scaled - defocus_U_pred_scaled))
    defocus_V_loss = tf.reduce_mean(tf.abs(defocus_V_true_scaled - defocus_V_pred_scaled))

    angle_loss = tf.reduce_mean(tf.abs(angle_true - angle_pred))
    image_loss = tf.reduce_mean(elementwise_losses)

    # Aggregate the elementwise losses
    aggregated_loss = image_loss + defocus_U_loss + defocus_V_loss + angle_loss

    return aggregated_loss


def exampleCTFApplyingFunction(df_metadata):
    defocusU = df_metadata.head(1)['DEFOCUS_U'].values[0]
    defocusV = df_metadata.head(1)['DEFOCUS_V'].values[0]
    kV = df_metadata.head(1)['kV'].values[0]
    defocusA = df_metadata.head(1)['Angle'].values[0]

    cs = 2.7e7
    sampling_rate = 1
    size = 512
    epsilon = 1e-8

    ctf_array_ts = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU, defocusV=defocusV, Cs=cs,
                                        phase_shift_PP=0, angle_ast=defocusA)

    ctf_array2_ts = call_ctf_function_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU, defocusV=defocusV, Cs=cs,
                                         phase_shift_PP=0, angle_ast=defocusA + 45)

    ctf_array = call_ctf_function(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU,
                                  defocusV=defocusV, Cs=cs,
                                  phase_shift_PP=0, angle_ast=defocusA)

    ctf_array2 = call_ctf_function(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU,
                                   defocusV=defocusV, Cs=cs,
                                   phase_shift_PP=0, angle_ast=defocusA + 45)
    def pearson_correlation_ts(array1, array2):
        # Flatten the arrays to make them 1D
        ctf_array_true_flat = tf.reshape(array1, [-1])
        ctf_array_pred_flat = tf.reshape(array2, [-1])

        # Calculate mean-centered vectors
        mean_true = tf.reduce_mean(ctf_array_true_flat)
        mean_pred = tf.reduce_mean(ctf_array_pred_flat)

        centered_true = ctf_array_true_flat - mean_true
        centered_pred = ctf_array_pred_flat - mean_pred

        # Calculate Pearson correlation coefficient
        numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))

        correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)

        return correlation_coefficient

    def pearson_correlation(array1, array2):
        from scipy.stats import pearsonr
        # Flatten the images into 1D arrays
        flat_image1 = array1.flatten()
        flat_image2 = array2.flatten()

        # Calculate Pearson correlation coefficient
        correlation_coefficient, p_value = pearsonr(flat_image1, flat_image2)

        return correlation_coefficient

    correlation_coefficient_TS = pearson_correlation_ts(ctf_array_ts, ctf_array2_ts)
    print(correlation_coefficient_TS)
    #
    correlation_coefficient = pearson_correlation(ctf_array, ctf_array2)
    print(correlation_coefficient)

    # Check if the values are approximately equal
    # if np.allclose(ctf_array,  ctf_array_ts.numpy(), rtol=1e-5, atol=1e-8):
    #     print("The CTF values from NumPy and TensorFlow functions are approximately equal.")
    # else:
    #     print("The CTF values from NumPy and TensorFlow functions are not equal.")

    # Plot the first image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ctf_array_ts.numpy(), cmap='gray')
    plt.title('Image 1')

    # Overlay the second image on top of the first
    plt.subplot(1, 2, 2)
    plt.imshow(ctf_array_ts.numpy() - ctf_array2_ts.numpy(), cmap='gray')  # Background image
    # plt.imshow(ctf_array2, cmap='viridis', alpha=0.5)  # Overlay image with some transparency
    plt.title('Difference')

    plt.show()

def extract_CNN_layer_features(modelDir, image_example_path, layers, defocus_scaler):
    one_image_data = centerWindow(image_example_path, objective_res=2, sampling_rate=1)
    one_image_data = one_image_data.reshape((-1, 256, 256, 1))

    model_defocus = DeepDefocusMultiOutputModel(width=256, height=256).getFullModel(learning_rate=0.001, defocus_scaler=defocus_scaler, cs=2.7e7, kV=200)
    model_defocus.load_weights(filepath=os.path.join(modelDir, 'Best_Weights'))
    features_path = os.path.join(modelDir, "featuresExtraction/")

    try:
        os.makedirs(features_path)
    except FileExistsError:
        pass

    model_defocus_layers = model_defocus.layers
    model_defocus_input = model_defocus.input

    layer_outputs_defocus = [layer.output for layer in model_defocus_layers]
    features_defocus_model = Model(inputs=model_defocus_input, outputs=layer_outputs_defocus)

    extracted_benchmark = features_defocus_model(one_image_data)

    # For the input image
    f1_benchmark = extracted_benchmark[0]
    print('\n Input benchmark shape:', f1_benchmark.shape)
    imgs = f1_benchmark[0, ...]
    plt.figure(figsize=(5, 5))
    plt.imshow(imgs[..., 0], cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(os.path.join(features_path, "features_layer_%s" % str(0)))
    # For the rest of the layers
    for layer in range(1, layers, 1):
        feature_benchmark = extracted_benchmark[layer]
        print('\n feature_benchmark shape:', feature_benchmark.shape)
        print('Layer ', layer)
        filters = feature_benchmark.shape[-1]
        imgs = feature_benchmark[0, ...]

        # Dynamically adjust the number of rows and columns based on the number of filters
        rows = 2
        cols = int(filters / 2) if filters % 2 == 0 else int(filters / 2) + 1
        # Dynamically adjust figsize based on the number of columns
        figsize = (cols * 5, rows * 5)

        plt.figure(figsize=figsize)
        for n in range(filters):
            ax = plt.subplot(rows, cols, n + 1)
            plt.imshow(imgs[..., n], cmap='gray')
            plt.axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(os.path.join(features_path, "features_layer_%s" % str(layer)))
        plt.close()


