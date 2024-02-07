import os
import sys
from time import time
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import startSessionAndInitialize, make_data_descriptive_plots, make_training_plots, prepareTestData, \
    make_testing_plots, make_testing_angle_plots, normalize_angle
from dataGenerator import CustomDataGenPINN
from DeepDefocusModel import DeepDefocusMultiOutputModel, extract_CNN_layer_features, exampleCTFApplyingFunction
import datetime
from sklearn.preprocessing import RobustScaler
import numpy as np

BATCH_SIZE = 16
EPOCHS = 300
TEST_SIZE = 0.10
LEARNING_RATE_DEF = 0.0001
SCALE_FACTOR = 50000

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
    input_size = (256, 256, 1)
    trainDefocus = True
    ground_Truth = False # True
    testing_Bool = True
    plots_Bool = True

    # ----------- INITIALIZING SYSTEM ------------------
    startSessionAndInitialize()

    # ----------- LOADING DATA ------------------
    print("Loading data...")
    path_metadata = os.path.join(metadataDir, "metadata.csv")
    df_metadata = pd.read_csv(path_metadata)

    # Stack the 'defocus_U' and 'defocus_V' columns into a single column
    stacked_data = df_metadata[[COLUMNS['defocus_U'], COLUMNS['defocus_V']]].stack().reset_index(drop=True)
    # Reshape the stacked data to a 2D array
    stacked_data_2d = stacked_data.values.reshape(-1, 1)

    # Instantiate the RobustScaler
    scaler = RobustScaler()
    # Fit the scaler to the stacked data
    scaler.fit(stacked_data_2d)
    df_metadata['DEFOCUS_U_SCALED'] = scaler.transform(df_metadata[COLUMNS['defocus_U']].values.reshape(-1, 1))
    df_metadata['DEFOCUS_V_SCALED'] = scaler.transform(df_metadata[COLUMNS['defocus_V']].values.reshape(-1, 1))

    df_metadata['NORMALIZED_ANGLE'] = df_metadata[COLUMNS['angle']]/180

    #scaler_factor = SCALE_FACTOR
    #df_metadata['DEFOCUS_U_SCALED'] = df_metadata[COLUMNS['defocus_U']]/scaler_factor
    #df_metadata['DEFOCUS_V_SCALED'] = df_metadata[COLUMNS['defocus_V']]/scaler_factor

    # ----------- STATISTICS ------------------
    print(df_metadata.describe())

    # ---------------- DESCRIPTIVE PLOTS ------------------------
    if plots_Bool:
        make_data_descriptive_plots(df_metadata, modelDir, COLUMNS, trainDefocus, ground_Truth)

    # ----------- SPLIT DATA: TRAIN, VALIDATE and TEST ------------
    df_training, df_test = train_test_split(df_metadata, test_size=TEST_SIZE)
    df_train, df_validate = train_test_split(df_training, test_size=0.20)

    # ----------- TRAINING MODELS-------------------
    if trainDefocus:
        # OJO: The number of batches is equal to len(df)//batch_size
        traingen = CustomDataGenPINN(data=df_train.head(7380), batch_size=BATCH_SIZE)

        valgen = CustomDataGenPINN(data=df_validate.head(1840), batch_size=BATCH_SIZE) # Probar batchsize 1

        testgen = CustomDataGenPINN(data=df_test, batch_size=1)

        path_logs_defocus = os.path.join(modelDir, "logs_defocus/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

        callbacks_list_def = [
            callbacks.CSVLogger(os.path.join(path_logs_defocus, 'defocus.csv'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=path_logs_defocus, histogram_freq=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=15),
            callbacks.ModelCheckpoint(filepath=os.path.join(path_logs_defocus, 'Best_Weights'),
                                      save_weights_only=True,
                                      save_best_only=True,
                                      monitor='val_loss',
                                      verbose=0)
            ]

        # ----------- TRAINING DEFOCUS MODEL-------------------
        # Check if GPUs are available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Create a MirroredStrategy
                strategy = tf.distribute.MirroredStrategy()

                with ((strategy.scope())):
                    # Define and compile your model within the strategy scope
                    print("Training defocus model")
                    start_time = time()
                    model_defocus = DeepDefocusMultiOutputModel(width=input_size[0], height=input_size[1]
                                                                ).getFullModel(learning_rate=LEARNING_RATE_DEF,
                                                                               defocus_scaler=scaler, cs=2.7e7,
                                                                               kV=200)

                # Train the model using fit method
                history_defocus = model_defocus.fit(traingen,
                                                    validation_data=valgen,
                                                    epochs=EPOCHS,
                                                    callbacks=callbacks_list_def,
                                                    verbose=1
                                                    )

                elapsed_time = time() - start_time
                print("Time in training model: %0.10f seconds." % elapsed_time)

                if plots_Bool:
                    make_training_plots(history_defocus, path_logs_defocus, "defocus_")

                one_image_data_path = df_test.head(1)['FILE'].values[0]
                extract_CNN_layer_features(path_logs_defocus, one_image_data_path, layers=7, defocus_scaler=scaler)
                #exampleCTFApplyingFunction(df_train)

            except Exception as e:
                print(e)
        else:
            print("No GPU devices available.")

    # ----------- SAVING DEFOCUS MODEL AND VAL INFORMATION -------------------
    # TODO: NOT FOR THE MOMENT
    # model.save(os.path.join(modelDir, 'model.h5'))

    # TODO THIS SHOULD BE IN ANOTHER SCRIPT
    # ----------- TESTING DEFOCUS MODEL -------------------
    if testing_Bool:
        print("Test mode")
        # loadModelDir = os.path.join(modelDir, 'model.h5')
        # model = load_model(loadModelDir)
        model_defocus = DeepDefocusMultiOutputModel(width=256, height=256).getFullModel(learning_rate=0.001,
                                                                                        defocus_scaler=scaler,
                                                                                        cs=2.7e7, kV=200)
        model_defocus.load_weights(filepath=os.path.join(path_logs_defocus, 'Best_Weights'))

        imagesTest, defocusTest, anglesTest = prepareTestData(df_test)

        if trainDefocus:
            print("Testing defocus model")
            defocusPrediction_scaled = model_defocus.predict(testgen)
            # Transform back
            defocusPrediction = np.zeros_like(defocusPrediction_scaled)
            defocusPrediction[:, 0] = scaler.inverse_transform(defocusPrediction_scaled[:, 0].reshape(-1, 1)).flatten()
            defocusPrediction[:, 1] = scaler.inverse_transform(defocusPrediction_scaled[:, 1].reshape(-1, 1)).flatten()
            #defocusPrediction[:, 0] = defocusPrediction_scaled[:, 0] * scaler_factor
            #defocusPrediction[:, 1] = defocusPrediction_scaled[:, 1] * scaler_factor
            defocusPrediction[:, 2] = defocusPrediction_scaled[:, 2] * 180

            mae_u = mean_absolute_error(defocusTest[:, 0], defocusPrediction[:, 0])
            print("Final mean absolute error defocus_U val_loss: ", mae_u)

            mae_v = mean_absolute_error(defocusTest[:, 1], defocusPrediction[:, 1])
            print("Final mean absolute error defocus_V val_loss: ", mae_v)

            mae_a = mean_absolute_error(anglesTest[:, 0], defocusPrediction[:, 2])
            print("Final mean absolute error angle val_loss: ", mae_a)


            mae_test_path = os.path.join(path_logs_defocus, "mae_test_results.txt")

            with open(mae_test_path, "w") as f:
                f.write("Final mean absolute error defocus_U val_loss: {}\n".format(mae_u))
                f.write("Final mean absolute error defocus_V val_loss: {}\n".format(mae_v))
                f.write("Final mean absolute error angle val_loss: {}\n".format(mae_a))

            print("Results written to mae_test_results.txt")

            if plots_Bool:
                make_testing_plots(defocusPrediction, defocusTest, path_logs_defocus)
                make_testing_angle_plots(defocusPrediction[:, 2], anglesTest, path_logs_defocus)

    exit(0)
