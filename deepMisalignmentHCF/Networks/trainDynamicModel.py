""" This module trains and validate the different models to solve the misalignment detection problem. """


import datetime
from time import time
import os
import sys

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

from CreateModel import compileModel, scratchModel, getCallbacks
from classes import DataGenerator
import plotUtils
import utils

SUBTOMO_SIZE = 32  # Dimensions of the subtomos (cubic, SUBTOMO_SIZE x SUBTOMO_SIZE x SUBTOMO_SIZE shape)
BATCH_SIZE = 64  # Number of boxes per batch
NUMBER_RANDOM_BATCHES = -1
EPOCHS = 50  # Number of epochs
LEARNING_RATE = 0.0001  # Learning rate
TESTING_SPLIT = 0.15  # Ratio of data used for testing
VALIDATION_SPLIT = 0.2  # Ratio of data used for validation

if __name__ == "__main__":
    # Running command

    # Check no program arguments missing
    if len(sys.argv) == 4:
        retrainModel = False
        print("Starting new model training mode")
    elif len(sys.argv) == 5:
        retrainModel = True
        print("Starting retraining model mode")

        # Path with the previously trained model
        pretrainedModelPath = sys.argv[4]
    else:
        print("Usage: scipion python trainDynamicModel.py <stackDir> <verboseOutput 0/1> <generatePlots 0/1> "
              "[modelDir]")
        sys.exit()

    # Path with the input stack of data
    stackDir = sys.argv[1]

    # Verbose output
    if sys.argv[2] == "0":
        verboseOutput = False
    elif sys.argv[2] == "1":
        verboseOutput = True
    else:
        raise Exception("Invalid option for <verboseOutput 0/1>. This option only accepts 0 or 1 input values.")

    # Generate output plots
    if sys.argv[3] == "0":
        generatePlots = False
    elif sys.argv[3] == "1":
        generatePlots = True
    else:
        raise Exception("Invalid option for <generatePlots 0/1>. This option only accepts 0 or 1 input values.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------------------------ PREPROCESS DATA
    print("Loading data...")
    start_time = time()
    inputSubtomoStreamAli = np.load(os.path.join(stackDir, "inputDataStreamAli.npy"))
    inputSubtomoStreamMisali = np.load(os.path.join(stackDir, "inputDataStreamMisali.npy"))

    numberOfAliSubtomos = len(inputSubtomoStreamAli)
    numberOfMisaliSubtomos = len(inputSubtomoStreamMisali)
    totalSubtomos = numberOfAliSubtomos + numberOfMisaliSubtomos

    # Normalize input subtomo data stream to N(0,1)
    normalizedInputSubtomoStreamAli = utils.normalizeInputDataStream(inputSubtomoStreamAli)
    normalizedInputSubtomoStreamMisali = utils.normalizeInputDataStream(inputSubtomoStreamMisali)

    # Test normalization
    # print("\n")
    # print("normalizedInputSubtomoStreamAli stats: mean " + str(np.mean(normalizedInputSubtomoStreamAli)) +
    #       " std: " + str(np.std(normalizedInputSubtomoStreamAli)))
    # print("normalizedInputSubtomoStreamMisali stats: mean " + str(np.mean(normalizedInputSubtomoStreamMisali)) +
    #       " std: " + str(np.std(normalizedInputSubtomoStreamMisali)))
    # print("\n")

    # ------------------------------------------------------------ PRODUCE SIDE INFO
    # Update the number of random batches respect to the dataset and batch sizes
    NUMBER_RANDOM_BATCHES = totalSubtomos // BATCH_SIZE

    # Output classes distribution info
    aliSubtomosRatio = numberOfAliSubtomos / totalSubtomos
    misaliSubtomosRatio = numberOfMisaliSubtomos / totalSubtomos

    if verboseOutput:
        print("\nClasses distribution:\n"
              "Aligned: %d (%.3f%%)\n"
              "Misaligned: %d (%.3f%%)\n\n"
              % (numberOfAliSubtomos, aliSubtomosRatio * 100,
                 numberOfMisaliSubtomos, misaliSubtomosRatio * 100))

    # Plot classes distribution info histogram
    if generatePlots:
        plotUtils.plotClassesDistributionDynamic(numberOfAliSubtomos, numberOfMisaliSubtomos)

    # ------------------------------------------------------------ DATA AUGMENTATION
    start_time = time()

    # Data augmentation for aligned subtomos
    generatedSubtomosAli = []

    print("Data augmentation for aligned subtomos:")

    for i in range(numberOfAliSubtomos):

        newSubtomos = utils.dataAugmentationSubtomoDynamic(normalizedInputSubtomoStreamAli[i, :, :, :],
                                                           (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE))

        for subtomo in newSubtomos:
            generatedSubtomosAli.append(subtomo)

    generatedSubtomosArrayAli = np.zeros((len(generatedSubtomosAli),
                                          SUBTOMO_SIZE,
                                          SUBTOMO_SIZE,
                                          SUBTOMO_SIZE),
                                         dtype=np.float64)

    for i, subtomo in enumerate(generatedSubtomosAli):
        generatedSubtomosArrayAli[i, :, :, :] = subtomo[:, :, :]

    print("Number of new subtomos generated: %d\n" % len(generatedSubtomosAli))

    print("Data structure shapes BEFORE augmentation:")
    print("Input aligned subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

    normalizedInputSubtomoStreamAli = np.concatenate((normalizedInputSubtomoStreamAli,
                                                      generatedSubtomosArrayAli))

    print("Data structure shapes AFTER augmentation:")
    print("Input misaligned subtomo stream: " + str(normalizedInputSubtomoStreamAli.shape))

    # Data augmentation for misaligned subtomos
    generatedSubtomosMisali = []

    print("Data augmentation for misaligned subtomos:")

    for i in range(numberOfMisaliSubtomos):

        newSubtomos = utils.dataAugmentationSubtomoDynamic(normalizedInputSubtomoStreamMisali[i, :, :, :],
                                                           (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE))

        for subtomo in newSubtomos:
            generatedSubtomosMisali.append(subtomo)

    generatedSubtomosArrayMisali = np.zeros((len(generatedSubtomosMisali),
                                             SUBTOMO_SIZE,
                                             SUBTOMO_SIZE,
                                             SUBTOMO_SIZE),
                                            dtype=np.float64)

    for i, subtomo in enumerate(generatedSubtomosMisali):
        generatedSubtomosArrayMisali[i, :, :, :] = subtomo[:, :, :]

    print("Number of new subtomos generated: %d\n" % len(generatedSubtomosMisali))

    print("Data structure shapes BEFORE augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamMisali.shape))

    normalizedInputSubtomoStreamMisali = np.concatenate((normalizedInputSubtomoStreamMisali,
                                                         generatedSubtomosArrayMisali))

    print("Data structure shapes AFTER augmentation:")
    print("Input subtomo stream: " + str(normalizedInputSubtomoStreamMisali.shape))

    dataAug_time = time() - start_time
    print("Time spent in data augmentation: %0.10f seconds.\n\n" % dataAug_time)

    # ------------------------------------------------------------ SPLIT DATA
    # Aligned subtomos
    normISSAli_train, normISSAli_test = train_test_split(normalizedInputSubtomoStreamAli,
                                                         test_size=TESTING_SPLIT,
                                                         random_state=42)

    # Misligned subtomos
    normISSMisali_train, normISSMisali_test = train_test_split(normalizedInputSubtomoStreamMisali,
                                                               test_size=TESTING_SPLIT,
                                                               random_state=42)

    print("Data objects final dimensions")
    print('Input train aligned subtomos matrix: ' + str(np.shape(normISSAli_train)))
    print('Input test aligned subtomos matrix: ' + str(np.shape(normISSAli_test)))
    print('Input train misaligned subtomos matrix: ' + str(np.shape(normISSMisali_train)))
    print('Input test misaligned subtomos matrix: ' + str(np.shape(normISSMisali_test)) + '\n')

    elapsed_time = time() - start_time
    print("Overall time spent preparing the data: %0.10f seconds.\n\n" % elapsed_time)

    # ------------------------------------------------------------ TRAIN MODEL
    print("Train model")
    start_time = time()

    # Date and time for output generation
    dateAndTime = str(datetime.datetime.now())
    dateAndTimeVector = dateAndTime.split(' ')
    dateAndTime = dateAndTimeVector[0] + "_" + dateAndTimeVector[1]
    dateAndTime = dateAndTime.replace(":", "-")

    # Validation/training ID toggle vectors
    aliID_validation, aliID_train = utils.generateTrainingValidationVectors(len(normISSAli_train), VALIDATION_SPLIT)
    misaliID_validation, misaliID_train = utils.generateTrainingValidationVectors(len(normISSMisali_train),
                                                                                  VALIDATION_SPLIT)

    # print("aliID_validation: " + str(len(aliID_validation)))
    # print(sorted(aliID_validation))
    # print("aliID_train: " + str(len(aliID_train)))
    # print(sorted(aliID_train))
    # print("misaliID_validation: " + str(len(misaliID_validation)))
    # print(sorted(misaliID_validation))
    # print("misaliID_train: " + str(len(misaliID_train)))
    # print(sorted(misaliID_train))

    # Parameters
    params = {'aliData': normISSAli_train,
              'misaliData': normISSMisali_train,
              'number_batches': NUMBER_RANDOM_BATCHES,
              'batch_size': BATCH_SIZE,
              'dim': (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE)}

    # Generators
    training_generator = DataGenerator(aliIDs=aliID_train,
                                       misaliIDs=misaliID_train,
                                       **params)
    validation_generator = DataGenerator(aliIDs=aliID_validation,
                                         misaliIDs=misaliID_validation,
                                         **params)

    # Compile model
    if not retrainModel:
        print("Generating a de novo model for training")
        model = compileModel(model=scratchModel(),
                             learningRate=LEARNING_RATE)
    else:
        print("Loading pretrained model located at : " + pretrainedModelPath)
        model = load_model(pretrainedModelPath)

    # Train model on dataset
    print("Training model...")

    dirPath = os.path.join(stackDir, "outputLog_" + dateAndTime)

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        use_multiprocessing=True,
                        workers=1,
                        callbacks=getCallbacks(dirPath))

    myValLoss = np.zeros(1)
    myValLoss[0] = history.history['val_loss'][-1]

    np.savetxt(os.path.join(dirPath, "model.txt"), myValLoss)
    model.save(os.path.join(dirPath, "model.h5"))

    elapsed_time = time() - start_time

    print("Time spent training the model: %0.10f seconds." % elapsed_time)

    if generatePlots:
        plotUtils.plotTraining(history, EPOCHS)

    # ------------------------------------------------------------ TEST MODEL
    print("\n\nTest model...\n")
    start_time = time()

    loadModelDir = os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.h5')
    model = load_model(loadModelDir)

    normISS_test, misalignmentInfoVector_test = utils.combineAliAndMisaliVectors(normISSAli_test,
                                                                                 normISSMisali_test,
                                                                                 SUBTOMO_SIZE,
                                                                                 shuffle=False)
    # Testing the numpy array generation
    #
    # import xmippLib as xmipp
    # inputSubtomoArray1 = xmipp.Image()
    # inputSubtomo1 = xmipp.Image()
    #
    # for i in range(len(normISS_test)):
    #     X_tmp = normISS_test[i, :]
    #     inputSubtomoArray1.setData(X_tmp)
    #     inputSubtomoArray1.write(str(i)+"_test_nparray_subtomo.mrc")

    # print("len(normISS_test) " + str(len(normISS_test)))
    # print("len(misalignmentInfoVector_test) " + str(len(misalignmentInfoVector_test)))

    misalignmentInfoVector_prediction = model.predict(normISS_test)

    print("misalignmentInfoVector_prediction")
    print(misalignmentInfoVector_prediction)
    print("misalignmentInfoVector_test")
    print(misalignmentInfoVector_test)

    # Convert the set of probabilities from the previous command into the set of predicted classes
    misalignmentInfoVector_predictionClasses = np.round(misalignmentInfoVector_prediction)

    print(misalignmentInfoVector_predictionClasses)

    np.savetxt(os.path.join(stackDir, 'model_prediction.txt'),
               misalignmentInfoVector_predictionClasses)

    elapsed_time = time() - start_time
    print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)

    mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
    print("Final model mean absolute error val_loss: %f\n" % mae)

    loss = model.evaluate(normISS_test,
                          misalignmentInfoVector_test,
                          verbose=2)

    print("Testing set mean absolute error: {:5.4f}".format(loss[0]))
    print("Testing set accuracy: {:5.4f}".format(loss[1]))

    # Plot results from testing
    if generatePlots:
        plotUtils.plotTesting(
            misalignmentInfoVector_test,
            misalignmentInfoVector_predictionClasses
        )