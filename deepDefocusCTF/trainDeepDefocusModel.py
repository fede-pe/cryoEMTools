import os
import sys
from time import time
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:XLA_GPU:0"
import tensorflow.keras.callbacks as callbacks
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import startSessionAndInitialize, getModelSeparatedDefocus, \
    getModelDefocusAngle, make_training_plots, prepareTestData, make_testing_plots, make_testing_angle_plots
from dataGenerator import CustomDataGen, CustomDataGenAngle
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 8 # Tiene que ser multiplos de tu tamaño de muestra
EPOCHS = 1
LEARNING_RATE = 0.001
IM_WIDTH = 512
IM_HEIGHT = 512
TEST_SIZE = 0.15

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
    input_size = (512, 512, 3) # TODO: IS THIS ALWAYS THE SAME?
    input_size_angle = (512, 512, 1) # TODO: IS THIS ALWAYS THE SAME?
    # This two condition should dissapear as both are going to be in the same model
    trainDefocus = True
    trainAngle = False
    testing_Bool = True
    plots_Bool = True

    # ----------- INITIALIZING SYSTEM ------------------
    startSessionAndInitialize()

    # ----------- LOADING DATA ------------------
    print("Loading data...")
    path_metadata = os.path.join(metadataDir, "metadata.csv")
    df_metadata = pd.read_csv(path_metadata)

    # ----------- STATISTICS ------------------
    print(df_metadata.describe())

    # ---------------- PLOTS ------------------------
    if plots_Bool:
        # HISTOGRAM
        df_defocus = df_metadata[[COLUMNS['defocus_U'], COLUMNS['defocus_V']]]
        df_defocus.plot.hist(alpha=0.5)
        plt.title('Defocus')
        plt.show()
        # BOXPLOT
        df_defocus.plot.box()
        plt.show()
        # SCATTER
        df_defocus.plot.scatter(x=COLUMNS['defocus_U'], y=COLUMNS['defocus_V'])
        plt.show()

        # TODO: more Angles plots
        # HISTOGRAM
        df_angle = df_metadata[[COLUMNS['angle'], COLUMNS['cosAngle'], COLUMNS['sinAngle']]]
        df_angle[COLUMNS['angle']].plot.hist(alpha=0.5)
        plt.title('Angle')
        plt.show()

    # ----------- SPLIT DATA: TRAIN, VALIDATE and TEST ------------
    # TODO: generate more data with the dataGenerator
    # DATA GENERATOR
    # print('Generating images...')
    # X_set_generated, Y_set_generated = data_generator(imagMatrix_Norm, defocusVector[:, :2])

    df_train, df_validate = train_test_split(df_metadata, test_size=0.20)
    _, df_test = train_test_split(df_metadata, test_size=0.10)

    # OJO: The number of batches is equal to len(df)//batch_size
    if trainDefocus:
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
                                batch_size=1, input_size=input_size) # BATCH_SIZE here is crucial since it needs to be a multiple of the len(df_test)

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

    # ----------- TRAINING MODELS-------------------
    print("Train mode")
    if trainDefocus:
        callbacks_list_def = [
            callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1.csv_defocus'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1_defocus'), histogram_freq=0,
                                  batch_size=BATCH_SIZE,
                                  write_graph=True, write_grads=False, write_images=False,
                                  embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None, embeddings_data=None),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=10)
            ]

        # ----------- TRAINING DEFOCUS MODEL-------------------
        print("Training defocus model")
        start_time = time()
        model = getModelSeparatedDefocus()
        history_defocus = model.fit(traingen,
                            validation_data=valgen,
                            epochs=EPOCHS,
                            callbacks=callbacks_list_def,
                            verbose=1,
                            )

        elapsed_time = time() - start_time
        print("Time in training model: %0.10f seconds." % elapsed_time)

    # ----------- SAVING DEFOCUS MODEL AND VAL INFORMATION -------------------
    # TODO: NOT FOR THE MOMENT
    # myValLoss = np.zeros(1)
    # myValLoss[0] = history.history['val_loss'][-1]
    # np.savetxt(os.path.join(modelDir, 'model.txt'), myValLoss)
    # model.save(os.path.join(modelDir, 'model.h5'))

    # ----------- TRAINING ANGLE MODEL-------------------
    if trainAngle:
        callbacks_list_ang = [
            callbacks.CSVLogger(os.path.join(modelDir, 'outCSV_06_28_1_angle.csv'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=os.path.join(modelDir, 'outTB_06_28_1_angle'), histogram_freq=0,
                                  batch_size=BATCH_SIZE,
                                  write_graph=True, write_grads=False, write_images=False,
                                  embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None, embeddings_data=None),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=10)
            ]
        print("Training defocus angle model")
        modelAngle = getModelDefocusAngle()
        history_angle = modelAngle.fit(trainAngen,
                                 validation_data=valAngen,
                                 epochs=EPOCHS,
                                 callbacks=callbacks_list_ang,
                                 verbose=1,
                                 )
    if plots_Bool:
        if trainDefocus:
            make_training_plots(history_defocus)
        if trainAngle:
            make_training_plots(history_angle)

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

        if trainAngle:
            print("Testing angle model")
            anglePrediction = modelAngle.predict(testAngen)  # Predict with the generator can be dangerous,
            # it needs to be a multiple of len(test)

            mae_sin = mean_absolute_error(anglesTest[:, 0], anglePrediction[:, 0])
            print("Final mean absolute error sinAng val_loss: ", mae_sin)

            mae_cos = mean_absolute_error(anglesTest[:, 1], anglePrediction[:, 1])
            print("Final mean absolute error cosAng val_loss: ", mae_cos)

        if plots_Bool:
            if trainDefocus:
                make_testing_plots(defocusPrediction, defocusTest)
            if trainAngle:
                make_testing_angle_plots(anglePrediction, anglesTest)

    exit(0)
