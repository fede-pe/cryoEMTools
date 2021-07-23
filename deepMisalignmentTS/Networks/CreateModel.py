
""" This module exposes the definition of the different models to be trained to solve the misalignment detection
problem. """

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Dropout, Flatten, Dense, \
    GlobalAveragePooling3D


def scratchModel():
    inputLayer = Input(shape=(32, 32, 32, 1), name="input")

    L = Conv3D(filters=8, kernel_size=3, activation="relu")(inputLayer)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=16, kernel_size=3, activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=32, kernel_size=3, activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=64, kernel_size=3, activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Flatten()(L)
    L = GlobalAveragePooling3D()(L)
    L = Dense(units=512, activation="relu")(L)
    L = Dropout(0.2)(L)

    L = Dense(units=6, name="output", activation="softmax")(L)

    return Model(inputLayer, L, name="3dDNNmisali")
