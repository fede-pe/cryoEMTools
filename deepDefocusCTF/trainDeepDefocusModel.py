import os
import sys
from time import time
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# os.environ["CUDA_VISIBLE_DEVICES"] = "/device:XLA_GPU:0"
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import startSessionAndInitialize, make_data_descriptive_plots, make_training_plots, prepareTestData, \
    make_testing_plots, make_testing_angle_plots
from dataGenerator import CustomDataGen, CustomDataGenAngle
from DeepDefocusModel import DeepDefocusMultiOutputModel
import datetime

BATCH_SIZE = 16
EPOCHS = 200
TEST_SIZE = 0.15
LEARNING_RATE_DEF = 0.0001
LEARNING_RATE_ANG = 0.001

COLUMNS = {'id': 'ID', 'defocus_U': 'DEFOCUS_U', 'defocus_V': 'DEFOCUS_V',
           'sinAngle': 'Sin(2*angle)', 'cosAngle': 'Cos(2*angle)',
           'angle': 'Angle', 'kV': 'kV', 'file': 'FILE'}


# ------------------------ MAIN PROGRAM -----------------------------
if __name__ == "__main__":
    # ----------- PARSING DATA -------------------
    if len(sys.argv) < 3:
        print("Usage: python3 trainDeepDefocusModel.py <metadataDir> <modelDir>")
        sys.exit()

    metadataDir = sys.argv[1]
    modelDir = sys.argv[2]
    input_size = (512, 512, 1)
    input_size_angle = (512, 512, 1)
    # This two condition should dissapear as both are going to be in the same model
    trainDefocus = True
    trainAngle = False
    ground_Truth = False
    testing_Bool = True
    plots_Bool = True

    # ----------- INITIALIZING SYSTEM ------------------
    startSessionAndInitialize()

    # ----------- LOADING DATA ------------------
    print("Loading data...")
    path_metadata = os.path.join(metadataDir, "metadata.csv")
    df_metadata = pd.read_csv(path_metadata)

    # Todo esto es solo por esta vez
    # old_path = '/home/dmarchan/DM/TFM/TestNewPhantomData/'
    # new_path = '/home/dmarchan/data_hilbert_tres/TestNewPhantomData'
    # df_metadata[COLUMNS['file']] = df_metadata[COLUMNS['file']].str.replace(old_path, new_path)
    # ----------- STATISTICS ------------------
    print(df_metadata.describe())

    # ---------------- DESCRIPTIVE PLOTS ------------------------
    if plots_Bool:
        make_data_descriptive_plots(df_metadata, modelDir, COLUMNS, trainDefocus, trainAngle, ground_Truth)

    # ----------- SPLIT DATA: TRAIN, VALIDATE and TEST ------------
    # TODO: generate more data with the dataGenerator
    # DATA GENERATOR
    # print('Generating images...')
    # X_set_generated, Y_set_generated = data_generator(imagMatrix_Norm, defocusVector[:, :2])

    df_train, df_validate = train_test_split(df_metadata, test_size=0.20)
    _, df_test = train_test_split(df_metadata, test_size=0.10)

    # ----------- TRAINING MODELS-------------------
    if trainDefocus:
        # OJO: The number of batches is equal to len(df)//batch_size
        traingen = CustomDataGen(df_train,
                                 X_col={'path': 'FILE'},
                                 y_col={'defocus_U': 'DEFOCUS_U', 'defocus_V': 'DEFOCUS_V'},
                                 batch_size=BATCH_SIZE, input_size=input_size)

        valgen = CustomDataGen(df_validate,
                               X_col={'path': 'FILE'},
                               y_col={'defocus_U': 'DEFOCUS_U', 'defocus_V': 'DEFOCUS_V'},
                               batch_size=BATCH_SIZE, input_size=input_size)

        testgen = CustomDataGen(df_test,
                                X_col={'path': 'FILE'},
                                y_col={'defocus_U': 'DEFOCUS_U', 'defocus_V': 'DEFOCUS_V'},
                                batch_size=1,
                                input_size=input_size)  # BATCH_SIZE here is crucial since it needs to be a multiple of the len(df_test)

        path_logs_defocus = os.path.join(modelDir, "logs_defocus/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

        callbacks_list_def = [
            callbacks.CSVLogger(os.path.join(path_logs_defocus, 'defocus.csv'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=path_logs_defocus, histogram_freq=1),
                                  # write_graph=True, write_grads=False, write_images=False,
                                  # embeddings_freq=0, embeddings_layer_names=None,
                                  # embeddings_metadata=None, embeddings_data=None),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=20)
            ]

        # ----------- TRAINING DEFOCUS MODEL-------------------
        print("Training defocus model")
        # Check if GPUs are available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Create a MirroredStrategy
                strategy = tf.distribute.MirroredStrategy()

                with strategy.scope():
                    # Define and compile your model within the strategy scope
                    print("Training defocus model")
                    start_time = time()
                    model = DeepDefocusMultiOutputModel().getModelSeparatedDefocus(learning_rate=LEARNING_RATE_DEF)

                # Train the model using fit method
                history_defocus = model.fit(traingen,
                                            validation_data=valgen,
                                            epochs=EPOCHS,
                                            callbacks=callbacks_list_def,
                                            verbose=1
                                            )
                elapsed_time = time() - start_time
                print("Time in training model: %0.10f seconds." % elapsed_time)

                if plots_Bool:
                    make_training_plots(history_defocus, path_logs_defocus, "defocus_")

            except Exception as e:
                print(e)
        else:
            print("No GPU devices available.")

    # ----------- SAVING DEFOCUS MODEL AND VAL INFORMATION -------------------
    # TODO: NOT FOR THE MOMENT
    # myValLoss = np.zeros(1)
    # myValLoss[0] = history.history['val_loss'][-1]
    # np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
    # model.save(os.path.join(modelDir, 'model.h5'))

    # ----------- TRAINING ANGLE MODEL-------------------
    if trainAngle:
        trainAngen = CustomDataGenAngle(df_train,
                                        X_col={'path': 'FILE'},
                                        y_col={'sinAngle': 'Sin(2*angle)', 'cosAngle': 'Cos(2*angle)'},
                                        batch_size=BATCH_SIZE, input_size=input_size_angle)

        valAngen = CustomDataGenAngle(df_train,
                                      X_col={'path': 'FILE'},
                                      y_col={'sinAngle': 'Sin(2*angle)', 'cosAngle': 'Cos(2*angle)'},
                                      batch_size=BATCH_SIZE, input_size=input_size_angle)

        testAngen = CustomDataGenAngle(df_test,
                                       X_col={'path': 'FILE'},
                                       y_col={'sinAngle': 'Sin(2*angle)', 'cosAngle': 'Cos(2*angle)'},
                                       batch_size=1,
                                       input_size=input_size_angle)  # BATCH_SIZE here is crucial since it needs to be a multiple of the len(df_test)

        path_logs_angle = os.path.join(modelDir, "logs_angle/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

        callbacks_list_ang = [
            callbacks.CSVLogger(os.path.join(path_logs_angle, 'angle.csv'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=path_logs_angle, histogram_freq=1),
                                  # write_graph=True, write_grads=False, write_images=False,
                                  # embeddings_freq=0, embeddings_layer_names=None,
                                  # embeddings_metadata=None, embeddings_data=None),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=10)
            ]

        # Check if GPUs are available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Create a MirroredStrategy
                strategy = tf.distribute.MirroredStrategy()

                with strategy.scope():
                    # Define and compile your model within the strategy scope
                    print("Training defocus angle model")
                    start_time = time()
                    modelAngle = DeepDefocusMultiOutputModel().getModelDefocusAngle(learning_rate=LEARNING_RATE_ANG)

                # Train the model using fit method
                history_angle = modelAngle.fit(trainAngen,
                                               validation_data=valAngen,
                                               epochs=EPOCHS,
                                               callbacks=callbacks_list_ang,
                                               verbose=1
                                               )

                elapsed_time = time() - start_time
                print("Time in training model: %0.10f seconds." % elapsed_time)

                if plots_Bool:
                    make_training_plots(history_angle, path_logs_angle, "angle_")

            except Exception as e:
                print(e)
        else:
            print("No GPU devices available.")

    # TODO THIS SHOULD BE IN ANOTHER SCRIPT
    # ----------- TESTING DEFOCUS MODEL -------------------
    if testing_Bool:
        print("Test mode")
        # loadModelDir = os.path.join(modelDir, 'model.h5')
        # model = load_model(loadModelDir)
        imagesTest, defocusTest, anglesTest = prepareTestData(df_test)
        if trainDefocus:
            print("Testing defocus model")
            defocusPrediction = model.predict(testgen)  # Predict with the generator can be dangerous,
            # it needs to be a multiple of len(test)

            mae_u = mean_absolute_error(defocusTest[:, 0], defocusPrediction[0])
            print("Final mean absolute error defocus_U val_loss: ", mae_u)

            mae_v = mean_absolute_error(defocusTest[:, 1], defocusPrediction[1])
            print("Final mean absolute error defocus_V val_loss: ", mae_v)

            if plots_Bool:
                make_testing_plots(defocusPrediction, defocusTest, path_logs_defocus)

        if trainAngle:
            print("Testing angle model")
            anglePrediction = modelAngle.predict(testAngen)  # Predict with the generator can be dangerous,
            # it needs to be a multiple of len(test)

            mae_sin = mean_absolute_error(anglesTest[:, 0], anglePrediction[0])
            print("Final mean absolute error sinAng val_loss: ", mae_sin)

            mae_cos = mean_absolute_error(anglesTest[:, 1], anglePrediction[1])
            print("Final mean absolute error cosAng val_loss: ", mae_cos)
            if plots_Bool:
                make_testing_angle_plots(anglePrediction, anglesTest, path_logs_angle)

    exit(0)
