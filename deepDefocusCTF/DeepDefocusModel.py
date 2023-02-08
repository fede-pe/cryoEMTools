import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Lambda, \
    Concatenate, Reshape
from tensorflow.keras import regularizers


class DeepDefocusMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one for defocus
    and another for the defocus angles. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """

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

    def build_defocus_branch_Fede(self, inputs):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        L = Conv2D(16, (15, 15), activation="relu")(inputs)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((3, 3))(L)
        L = Conv2D(16, (9, 9), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(16, (5, 5), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        return L

# In reality we can have only one build_defocus_branch General so you use it but with different filters size

    def build_defocus_branch(self, inputs, Xdim, factor):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = Conv2D(16, (int(Xdim*factor), int(Xdim*factor)), activation="relu", padding="valid")(inputs) # The smalles of the three (12,12)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((3, 3))(L)
        Xconv1dim = np.shape(L)[1]
        L = Conv2D(8, (int(Xconv1dim/10), int(Xconv1dim/10)), activation="relu", padding="valid")(L)
        # L = Conv2D(8, (6, 6), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        Xconv2dim = np.shape(L)[1]
        L = Conv2D(4, (int(Xconv2dim/10), int(Xconv2dim/10)), activation="relu", padding="valid")(L)
        # L = Conv2D(4, (3, 3), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        Xconv3dim = np.shape(L)[1]
        L = Conv2D(2, (int(Xconv3dim/10), int(Xconv3dim/10)), activation="relu", padding="valid")(L)
        # L = Conv2D(2, (3, 3), activation="relu", padding="valid")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Flatten()(L)
        # L = Dropout(0.1)(L)

        return L


    def build_defocus_angle_branch(self, input, Xdim, factor):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        L = Conv2D(16, (int(Xdim*factor), int(Xdim*factor)), activation="relu", padding="valid")(input)  # The biggest of the three (20, 20)
        L = BatchNormalization()(L)  # It is used for improving the speed, performance and stability
        L = MaxPooling2D((2, 2))(L)
        Xconv1dim = np.shape(L)[1]
        print(Xconv1dim)
        L = Conv2D(8, (int(Xconv1dim / 10), int(Xconv1dim / 10)), activation="relu", padding="valid")(L)
        # L = Conv2D(8, (15, 15), activation="relu", padding="valid")(L) OLD
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        Xconv2dim = np.shape(L)[1]
        L = Conv2D(4, (int(Xconv2dim / 10), int(Xconv2dim / 10)), activation="relu", padding="valid")(L)
        # L = Conv2D(4, (7, 7), activation="relu", padding="valid")(L) OLD
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        # Xconv3dim = np.shape(L)[1]
        # L = Conv2D(2, (int(Xconv3dim / 10), int(Xconv3dim / 10)), activation="relu", padding="valid")(L)
        # L = Conv2D(2, (3, 3), activation="relu", padding="valid")(L)
        # L = BatchNormalization()(L)
        # L = MaxPooling2D()(L)
        L = Flatten()(L)

        return L

    def assemble_model_defocus(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')

        input1 = Reshape((height, width, 1))(inputs[:, :, :, 0])
        input2 = Reshape((height, width, 1))(inputs[:, :, :, 1])
        input3 = Reshape((height, width, 1))(inputs[:, :, :, 2])

        defocus_branch_at_1 = self.build_defocus_branch(input1, height, factor=0.0586) # AT 1A
        defocus_branch_at_2 = self.build_defocus_branch(input2, height, factor=0.1172) # At 2A
        defocus_branch_at_3 = self.build_defocus_branch(input3, height, factor=0.176)  # At 3A
        concatted = Concatenate()([defocus_branch_at_1, defocus_branch_at_2, defocus_branch_at_3])

        L = Flatten()(concatted)
        L = Dropout(0.3)(L) #Este dropout es muy heavy
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(L)
        #L = Dropout(0.1)(L) #ESTO QUITARLO SI METE MUCHO DROPOUT
        #L = Dense(16, activation='relu')(L)
        # L = Dropout(0.2)(L)
        # L = Dense(32, activation='relu')(L)
        # L = Dropout(0.2)(L)
        L = Dense(2, activation='linear', name='defocus_output')(L)

        model = Model(inputs=inputs, outputs=[L],
                      name="deep_defocus_net")

        return model


    # NEW MODEL APPROACH SEPARATED DEFOCUS U and V branches
    def assemble_model_separated_defocus(self, width, height):
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

    def build_defocusU_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        #L = Dropout(0.1)(L) #ESTO QUITARLO SI METE MUCHO DROPOUT
        #L = Dense(16, activation='relu')(L)
        # L = Dropout(0.2)(L)
        # L = Dense(32, activation='relu')(L)
        # L = Dropout(0.2)(L)
        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)
        return defocusU

    def build_defocusV_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(convLayer)
        #L = Dropout(0.1)(L) #ESTO QUITARLO SI METE MUCHO DROPOUT
        #L = Dense(16, activation='relu')(L)
        # L = Dropout(0.2)(L)
        # L = Dense(32, activation='relu')(L)
        # L = Dropout(0.2)(L)
        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)
        return defocusV

    def assemble_model_angle(self, height, width):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 1)
        input = Input(shape=input_shape, name='input')
        input1 = Reshape((height, width, 1))(input)
        defocus_angles_branch = self.build_defocus_angle_branch(input1, height, 0.9)
        L = Flatten()(defocus_angles_branch)
        L = Dropout(0.2)(L)
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01))(L)
        L = Dense(2, activation='linear', name='defocus_angle_output')(L)

        model = Model(inputs=input, outputs=[L],
                      name="deep_defocus_angle_net")

        return model

    def assemble_model_defocus_Fede(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')
        defocus_branch = self.build_defocus_branch_Fede(inputs)
        L = Flatten()(defocus_branch)
        L = Dense(256, activation='relu')(L)
        L = Dropout(0.2)(L)
        L = Dense(2, activation='linear', name='defocus_output')(L)

        model = Model(inputs=inputs, outputs=[L],
                      name="deep_defocus_net_Fede")

        return model


    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')
        defocus_branch = self.build_defocus_branch_Fede(inputs)
        defocus_angles_branch = self.build_defocus_angle_branch(inputs)
        # concatted = Concatenate()([defocus_branch, defocus_angles_branch])
        model = Model(inputs=inputs, outputs=[defocus_branch, defocus_angles_branch],
                      name="deep_defocus_net")

        return model
