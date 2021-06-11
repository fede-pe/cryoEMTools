from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation, Add, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks


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
        x = Conv2D(filters=16, kernel_size=(12, 12), padding="same", activation='relu')(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (9, 9), padding="same", activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        return x


    def build_defocus_U_branch(self, inputs):
        """
        Used to build the defocus in U and V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear', name='defocus_U_output')(x)

        return x

    def build_defocus_V_branch(self, inputs):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear', name='defocus_V_output')(x)

        return x


    def build_defocus_Cosangle_branch(self, inputs):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear', name='defocus_Cosangles_output')(x)

        return x

    def build_defocus_Sinangle_branch(self, inputs):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear', name='defocus_Sinangles_output')(x)

        return x


    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')
        defocus_U_branch = self.build_defocus_U_branch(inputs)
        defocus_V_branch = self.build_defocus_V_branch(inputs)
        defocus_Cosangles_branch = self.build_defocus_Cosangle_branch(inputs)
        defocus_Sinangles_branch = self.build_defocus_Sinangle_branch(inputs)


        model = Model(inputs=inputs, outputs=[defocus_U_branch, defocus_V_branch,
                                              defocus_Cosangles_branch, defocus_Sinangles_branch],
                      name="deep_defocus_net")

        return model


    def build_defocus_branch(self, inputs):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(2, activation='linear', name='defocus_output')(x)

        return x


    def build_defocus_angle_branch(self, inputs):
        """
        Used to build the angle branch (cos and sin) of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(2, activation='linear', name='defocus_angles_output')(x)

        return x

    def assemble_full_model_original(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape, name='input')
        defocus_branch = self.build_defocus_branch(inputs)
        defocus_angles_branch = self.build_defocus_angle_branch(inputs)

        #concatted = Concatenate()([defocus_branch, defocus_angles_branch])

        model = Model(inputs=inputs, outputs=[defocus_branch, defocus_angles_branch],
                      name="deep_defocus_net")

        return model