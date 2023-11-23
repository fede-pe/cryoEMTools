import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D,\
    Lambda, Concatenate, Reshape, UpSampling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def angle_error_metric(y_true, y_pred):
    # Extract predicted sin and cos values from the model's output
    pred_sin = y_pred[0]
    pred_cos = y_pred[1]
    # Calculate the predicted angle
    pred_angle = tf.atan2(pred_sin, pred_cos)
    # Extract true sin and cos values from the ground truth
    true_sin = y_true[0]
    true_cos = y_true[1]
    # Calculate the true angle
    true_angle = tf.atan2(true_sin, true_cos)
    # Calculate the absolute error in radians
    angle_error = K.abs(pred_angle - true_angle)
    # Convert the angle error to degrees if needed
    angle_error_degrees = angle_error * (180 / np.pi)
    # Return the mean angle error
    return K.mean(angle_error_degrees)


class DeepDefocusMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one for defocus
    and another for the defocus angles. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def __init__(self, width=512, height=512):
        self.IM_WIDTH = width
        self.IM_HEIGHT = height

    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:

        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu')(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        return x

    def build_defocus_branch(self, input, Xdim, factor):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = Conv2D(32, (int(Xdim*factor), int(Xdim*factor)), activation="relu", padding="valid")(input)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((3, 3))(L)

        Xconv1dim = np.shape(L)[1]

        L = Conv2D(16, (int(Xconv1dim/10), int(Xconv1dim/10)), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        Xconv2dim = np.shape(L)[1]

        L = Conv2D(8, (int(Xconv2dim/10), int(Xconv2dim/10)), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        Xconv3dim = np.shape(L)[1]

        # L = Conv2D(4, (int(Xconv3dim/10), int(Xconv3dim/10)), activation="relu", padding="valid")(L)
        L = Conv2D(4, (3, 3), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        L = Flatten()(L)

        return L

    def build_defocus_angle_branch(self, input, Xdim, factor):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = Conv2D(16, (int(Xdim*factor), int(Xdim*factor)), activation="relu", padding="valid")(input)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((2, 2))(L)

        Xconv1dim = np.shape(L)[1]

        L = Conv2D(8, (int(Xconv1dim / 10), int(Xconv1dim / 10)), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        Xconv2dim = np.shape(L)[1]

        L = Conv2D(4, (int(Xconv2dim / 10), int(Xconv2dim / 10)), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        # L = Conv2D(2, (int(Xconv3dim / 10), int(Xconv3dim / 10)), activation="relu", padding="valid")(L)
        # L = Conv2D(2, (3, 3), activation="relu", padding="valid")(L)
        # L = BatchNormalization()(L)
        # L = MaxPooling2D()(L)

        L = Flatten()(L)

        return L

    def build_defocusU_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        L = Dropout(0.1)(L)
        L = Dense(16, activation='relu')(L)
        L = Dropout(0.1)(L)
        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)
        return defocusU

    def build_defocusV_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        L = Dropout(0.1)(L)
        L = Dense(16, activation='relu')(L)
        L = Dropout(0.1)(L)
        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)
        return defocusV

    def build_sin_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        sinA = Dense(1, activation='linear', name='sinAngle_output')(L)
        return sinA

    def build_cos_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        cosA = Dense(1, activation='linear', name='cosAngle_output')(L)
        return cosA

    def assemble_model_separated_defocus(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 1)
        input_layer = Input(shape=input_shape, name='input')

        defocus_branch_at_2 = self.build_defocus_branch(input_layer, height, factor=0.25)

        L = Dropout(0.1)(defocus_branch_at_2)

        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)

        model = Model(inputs=input_layer, outputs=[defocusU, defocusV],
                      name="deep_separated_defocus_net")

        return model

    def assemble_model_separated_defocus_branches(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')

        input1 = Reshape((height, width, 1))(inputs[:, :, :, 0])
        input2 = Reshape((height, width, 1))(inputs[:, :, :, 1])
        input3 = Reshape((height, width, 1))(inputs[:, :, :, 2])

        defocus_branch_at_1 = self.build_defocus_branch(input1, height, factor=0.0586) # At 1A
        defocus_branch_at_2 = self.build_defocus_branch(input2, height, factor=0.1172) # At 2A
        defocus_branch_at_3 = self.build_defocus_branch(input3, height, factor=0.176) # At 3A
        concatted = Concatenate()([defocus_branch_at_1, defocus_branch_at_2, defocus_branch_at_3])

        L = Flatten()(concatted)
        L = Dropout(0.3)(L) #Este dropout es muy heavy
        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)

        model = Model(inputs=inputs, outputs=[defocusU, defocusV],
                      name="deep_separated_defocus_net")

        return model

    def assemble_model_angle(self, height, width):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 1)
        input = Input(shape=input_shape, name='input')

        defocus_angles_branch = self.build_defocus_angle_branch(input, height, factor=0.9)

        L = Dropout(0.2)(defocus_angles_branch)

        sin_branch = self.build_sin_branch(L)
        cos_branch = self.build_cos_branch(L)

        model = Model(inputs=input, outputs=[sin_branch, cos_branch],
                      name="deep_defocus_angle_net")

        return model

    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')
        defocus_branch = self.build_defocus_branch(inputs)
        defocus_angles_branch = self.build_defocus_angle_branch(inputs)
        # concatted = Concatenate()([defocus_branch, defocus_angles_branch])
        model = Model(inputs=inputs, outputs=[defocus_branch, defocus_angles_branch],
                      name="deep_defocus_net")

        return model

    # ----------- GET MODEL -------------------
    def getFullModel(self, learning_rate):
        model = self.assemble_full_model(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()
        # plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss={'defocus_output': 'mae',
                            'defocus_angles_output': 'mae'},
                      loss_weights=None,
                      metrics={})  # 'defocus_output': 'msle',
        # 'defocus_angles_output': 'msle'})

        return model

    def getModelSeparatedDefocus(self, learning_rate):
        model = self.assemble_model_separated_defocus(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()
        # plot_model(model, to_file='"deep_defocus_net.png', show_shapes=True)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss={'defocus_U_output': 'mae',
                            'defocus_V_output': 'mae'},
                      loss_weights=None,
                      metrics={})

        return model

    def getModelDefocusAngle(self, learning_rate):
        model = self.assemble_model_angle(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      # loss={'defocus_angle_output': 'mae'},
                      loss={'sinAngle_output': 'mae',
                            'cosAngle_output': 'mae'},
                      loss_weights=None,
                      metrics=[angle_error_metric])  # TODO : This metric should be align with the mae have a look

        return model


class Autoencoder:
    def __init__(self, input_shape=(512, 512, 1), latent_dim=8):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = Input(shape=self.input_shape)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder part
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder

    def train_autoencoder(self, data_generator, epochs=50):
        self.autoencoder.fit(data_generator, epochs=epochs)

    def extract_features(self, input_data):
        return self.autoencoder.predict(input_data)