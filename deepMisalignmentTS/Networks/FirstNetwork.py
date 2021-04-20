#!/usr/bin/env python2
import numpy as np
import os
import sys
from time import time

batch_size = 128  # Number of boxes per batch

if __name__ == "__main__":

    # Check no program arguments missing
    if len(sys.argv) < 3:
        print("Usage: scipion python batch_deepDefocus.py <stackDir> <modelDir>")
        sys.exit()
    stackDir = sys.argv[1]
    modelDir = sys.argv[2]

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import keras.callbacks as callbacks
    from keras.models import Model, load_model
    from keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Dropout, Flatten, Dense
    from keras.optimizers import Adam


    def constructModel():
        inputLayer = Input(shape=(32, 32, 32, 1), name="input")

        x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputLayer)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x)

        outputs = Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = Model(inputLayer, outputs, name="3dcnn")
        return model

        # L = Conv3D(filters=32, kernel_size=8, activation="relu")(inputLayer)
        # L = MaxPool3D(pool_size=2)(L)
        # L = BatchNormalization()(L)
        #
        # L = Conv3D(filters=32, kernel_size=8, activation="relu")(L)
        # L = MaxPool3D(pool_size=2)(L)
        # L = BatchNormalization()(L)
        #
        # L = Conv3D(filters=32, kernel_size=8, activation="relu")(L)
        # L = MaxPool3D(pool_size=2)(L)
        # L = BatchNormalization()(L)
        #
        # L = Flatten()(L)
        # L = Dense(units=512, activation="relu")(L)
        # L = Dropout(0.2)(L)
        #
        # L = Dense(units=6, name="output", activation="softmax")(L)
        #
        # return Model(inputLayer, L, name="3dDNNmisali")



    model = constructModel()
    model.summary()

    print("Loading data...")
    start_time = time()
    inputSubtomoStream = np.load(os.path.join(stackDir, "inputDataStream.npy"))
    misalignmentInfoVector = np.load(os.path.join(stackDir, "misalignmentInfoList.npy"))
    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

    print("Train model")
    start_time = time()
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    callbacks_list = [callbacks.CSVLogger("./outCSV_06_28_1", separator=',', append=False),
                      callbacks.TensorBoard(log_dir='./outTB_06_28_1',
                                            histogram_freq=0,
                                            batch_size=128,
                                            write_graph=True,
                                            write_grads=False,
                                            write_images=False,
                                            embeddings_freq=0,
                                            embeddings_layer_names=None,
                                            embeddings_metadata=None,
                                            embeddings_data=None),
                      callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.1,
                                                  patience=5,
                                                  verbose=1,
                                                  mode='auto',
                                                  min_delta=0.0001,
                                                  cooldown=0,
                                                  min_lr=0)]

    history = model.fit(inputSubtomoStream,
                        misalignmentInfoVector,
                        batch_size=128,
                        epochs=100,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks_list)

    myValLoss = np.zeros(1)
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
    model.save(os.path.join(modelDir, 'model.h5'))
    elapsed_time = time() - start_time
    print("Time spent training the model: %0.10f seconds." % elapsed_time)

    print("Test model")
    start_time = time()
    loadModelDir = os.path.join(modelDir, 'model.txt')
    model = load_model(loadModelDir)
    imagPrediction = model.predict(inputSubtomoStream)
    np.savetxt(os.path.join(stackDir, 'imagPrediction.txt'), imagPrediction)
    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds." % elapsed_time)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(misalignmentInfoVector, imagPrediction)
    print("Final model mean absolute error val_loss: %f", mae)
