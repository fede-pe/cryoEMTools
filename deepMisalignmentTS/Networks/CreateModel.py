
""" This module contains the definition of the different models to be trained to solve the misalignment detection
problem. """
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Dropout, Dense, \
    GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def compileModel(model, learningRate):
    optimizer = Adam(lr=learningRate)

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(from_logits=False),  # loss='mean_absolute_error'
                  metrics=['accuracy'])

    return model


def scratchModel():
    inputLayer = Input(shape=(32, 32, 32, 1), name="input")

    L = Conv3D(filters=16, kernel_size=3, data_format="channels_last", padding="same", activation="relu")(inputLayer)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=16, kernel_size=3, data_format="channels_last", padding="same", activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=32, kernel_size=3, data_format="channels_last", padding="same", activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = Conv3D(filters=64, kernel_size=3, data_format="channels_last", padding="same", activation="relu")(L)
    L = MaxPool3D(pool_size=2)(L)
    L = BatchNormalization()(L)

    L = GlobalAveragePooling3D()(L)

#    L = Flatten()(L)
    L = Dense(units=512, activation="relu")(L)
    # L = Dense(units=256, activation="relu")(L)
    # L = Dense(units=128, activation="relu")(L)
    L = Dropout(0.2)(L)

    L = Dense(units=1, name="output", activation="sigmoid")(L)

    return Model(inputLayer, L, name="3dDNNmisali")


def getCallbacks(modelCheckpointFilePath):
    myCallbacks = []

    mcp = ModelCheckpoint(
        os.path.join(modelCheckpointFilePath, 'model.{epoch:02d}-{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_weights_only=False,
        save_best_only=True,
        mode='min',  # Because we are monitoring val_loss
    )

    myCallbacks.append(mcp)

    rlrop = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    )

    myCallbacks.append(rlrop)

    return myCallbacks
