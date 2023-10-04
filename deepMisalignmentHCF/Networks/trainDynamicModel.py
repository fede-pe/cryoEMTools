""" This module trains and validate the different models to solve the misalignment detection problem. """

import argparse
import datetime
from time import time
import os
import glob

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

from CreateModel import compileModel, scratchModel, getCallbacks
from classes import DataGenerator
import plotUtils
import utils

# Module variables
SUBTOMO_SIZE = 32  # Dimensions of the subtomos (cubic, SUBTOMO_SIZE x SUBTOMO_SIZE x SUBTOMO_SIZE shape)
BATCH_SIZE = 64  # Number of boxes per batch
NUMBER_RANDOM_BATCHES = -1
EPOCHS = 50  # Number of epochs
LEARNING_RATE = 0.0001  # Learning rate
TESTING_SPLIT = 0.15  # Ratio of data used for testing
VALIDATION_SPLIT = 0.2  # Ratio of data used for validation


class TrainDynamicModel:
    """ This class holds all the methods needed to train dynamically a DNN model """

    def __init__(self, stackDir, verboseOutput, generatePlots, normalize, modelDir):
        self.stackDir = stackDir
        self.verboseOutput = verboseOutput
        self.generatePlots = generatePlots
        self.normalize = normalize
        self.modelDir = modelDir

        if modelDir is None:
            self.retrainModel = False
        else:
            self.retrainModel = True

        # Side information variables
        self.totalSubtomos = 0
        self.totalAliSubtomos = 0
        self.totalMisaliSubtomos = 0

        # Dictionaries with dataset information
        self.aliDict = {}
        self.misaliDict = {}

        # Compose output folder
        dateAndTime = str(datetime.datetime.now())
        dateAndTimeVector = dateAndTime.split(' ')
        dateAndTime = dateAndTimeVector[0] + "_" + dateAndTimeVector[1]
        dateAndTime = dateAndTime.replace(":", "-")
        self.dirPath = os.path.join(stackDir, "outputLog_" + dateAndTime)

        if not os.path.exists(self.dirPath):
            os.makedirs(self.dirPath)

        # Trigger program execution
        self.produceSideInfo()
        if normalize:
            self.normalizeData()
        self.dataAugmentation()
        self.splitData()

    def produceSideInfo(self):
        """ Produce read input data arrays and metadata information """

        print("------------------------------------------ Data loading and side info generation")
        start_time = time()

        # Search for all available dataset
        inputDataArrays = glob.glob(f"{self.stackDir}/*_*.npy")

        for ida in inputDataArrays:
            fnNoExt = os.path.splitext(os.path.basename(ida))[0]
            [key, flag] = fnNoExt.split('_')

            if flag == "Ali":
                data = np.load(ida)
                self.aliDict[key] = (data, len(data))
                self.totalAliSubtomos += len(data)
                self.totalSubtomos += len(data)

            elif flag == "Misali":
                data = np.load(ida)
                self.misaliDict[key] = (data, len(data))
                self.totalMisaliSubtomos += len(data)
                self.totalSubtomos += len(data)

        # Update the number of random batches respect to the dataset and batch sizes
        NUMBER_RANDOM_BATCHES = self.totalSubtomos // BATCH_SIZE

        # Output classes distribution info
        aliSubtomosRatio = self.totalAliSubtomos / self.totalSubtomos
        misaliSubtomosRatio = self.totalMisaliSubtomos / self.totalSubtomos

        if self.verboseOutput:
            print("\nClasses distribution:")

            for key in self.aliDict.keys():
                totalAli = self.aliDict[key][1]
                totalMisali = self.misaliDict[key][1]
                total = totalAli + totalMisali

                print("Dataset %s:\t"
                      "Aligned: %d (%.3f%%)\t"
                      "Misaligned: %d (%.3f%%)\t"
                      % (key,
                         self.aliDict[key][1], (totalAli / total) * 100,
                         self.misaliDict[key][1], (totalMisali / total) * 100))

            print("\nSet of datasets:\t"
                  "Aligned: %d (%.3f%%)\t"
                  "Misaligned: %d (%.3f%%)\t"
                  % (self.totalAliSubtomos, aliSubtomosRatio * 100,
                     self.totalMisaliSubtomos, misaliSubtomosRatio * 100))

        # Plot classes distribution info histogram
        if self.generatePlots:
            plotUtils.plotClassesDistributionDynamic(self.aliDict, self.misaliDict, self.dirPath)

        sideInfo_time = time() - start_time
        print("Time spent in side information generation: %0.10f seconds.\n\n" % sideInfo_time)

    def normalizeData(self):
        """ Normalize given data array """

        print("------------------------------------------ Data normalization")
        start_time = time()

        for key in self.aliDict.keys():
            # Normalize input subtomo data stream to N(0,1)
            self.aliDict[key] = (utils.normalizeInputDataStream(self.aliDict[key][0]), self.aliDict[key][1])
            self.misaliDict[key] = (utils.normalizeInputDataStream(self.misaliDict[key][0]), self.misaliDict[key][1])

            # Test normalization
            if self.verboseOutput:
                print("Dataset %s:\t"
                      "Aligned: mean %s, std %s\t"
                      "Misaligned: mean %s, std %s\t"
                      % (key,
                         str(np.mean(self.aliDict[key][0])), str(np.std(self.aliDict[key][0])),
                         str(np.mean(self.misaliDict[key][0])), str(np.std(self.misaliDict[key][0]))))

        norm_time = time() - start_time
        print("Time spent in data normalization: %0.10f seconds.\n\n" % norm_time)

    def dataAugmentation(self):
        """ Method to perform data augmentation strategies to input data """

        print("------------------------------------------ Data augmentation")
        start_time = time()

        for key in self.aliDict.keys():
            numberOfAliSubtomos = self.aliDict[key][1]
            numberOfMisaliSubtomos = self.misaliDict[key][1]

            inputSubtomoStreamAli = self.aliDict[key][0]
            inputSubtomoStreamMisali = self.misaliDict[key][0]

            # Data augmentation for aligned subtomos
            generatedSubtomosAli = []

            for i in range(numberOfAliSubtomos):
                newSubtomos = utils.dataAugmentationSubtomoDynamic(inputSubtomoStreamAli[i, :, :, :],
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

            if self.verboseOutput:
                print("Number of new subtomos generated: %d\n" % len(generatedSubtomosAli))
                print("Data structure shapes BEFORE augmentation:")
                print("Input aligned subtomo stream: " + str(inputSubtomoStreamAli.shape))

            inputSubtomoStreamAli = np.concatenate((inputSubtomoStreamAli,
                                                    generatedSubtomosArrayAli))

            self.aliDict[key] = (inputSubtomoStreamAli, numberOfAliSubtomos)

            if self.verboseOutput:
                print("Data structure shapes AFTER augmentation:")
                print("Input misaligned subtomo stream: " + str(inputSubtomoStreamAli.shape))

            # Data augmentation for misaligned subtomos
            generatedSubtomosMisali = []

            for i in range(numberOfMisaliSubtomos):

                newSubtomos = utils.dataAugmentationSubtomoDynamic(inputSubtomoStreamMisali[i, :, :, :],
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

            if self.verboseOutput:
                print("Number of new subtomos generated: %d\n" % len(generatedSubtomosMisali))
                print("Data structure shapes BEFORE augmentation:")
                print("Input subtomo stream: " + str(inputSubtomoStreamMisali.shape))

            inputSubtomoStreamMisali = np.concatenate((inputSubtomoStreamMisali,
                                                       generatedSubtomosArrayMisali))

            self.misaliDict[key] = (inputSubtomoStreamMisali, numberOfMisaliSubtomos)

            if self.verboseOutput:
                print("Data structure shapes AFTER augmentation:")
                print("Input subtomo stream: " + str(inputSubtomoStreamMisali.shape))

        dataAug_time = time() - start_time
        print("Time spent in data augmentation: %0.10f seconds.\n\n" % dataAug_time)

    def splitData(self):
        """ Method to split data into train and test"""

        print("------------------------------------------ Data split train-test")
        start_time = time()

        for key in self.aliDict.keys():
            # Aligned subtomos
            aliTrain, aliTest = train_test_split(self.aliDict[key],
                                                 test_size=TESTING_SPLIT,
                                                 random_state=42)

            self.aliDict[key] = (aliTrain, aliTest)

            # Misligned subtomos
            misaliTrain, misaliTest = train_test_split(self.misaliDict[key],
                                                       test_size=TESTING_SPLIT,
                                                       random_state=42)

            self.misaliDict[key] = (misaliTrain, misaliTest)

            if self.verboseOutput:
                print("Data objects final dimensions for dataset %s" % key)
                print('Input train aligned subtomos matrix: ' + str(np.shape(aliTrain)))
                print('Input test aligned subtomos matrix: ' + str(np.shape(aliTest)))
                print('Input train misaligned subtomos matrix: ' + str(np.shape(aliTrain)))
                print('Input test misaligned subtomos matrix: ' + str(np.shape(aliTest)) + '\n')

            elapsed_time = time() - start_time
            print("Overall time spent in data train-test splitting: %0.10f seconds.\n\n" % elapsed_time)


# ----------------------------------- Main ------------------------------------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Read params
    description = "Script for training a DDN to detect alignment errors in tomography based on artifacted fiducial " \
                  "markers\n"

    parser = argparse.ArgumentParser(description=description)

    # Define the mandatory input path parameters
    parser.add_argument('--stackDir', required=True, help='Input path to folder containing two numpy arrays containing'
                                                          'both aligned and misaligned fiducials (mandatory)')

    # Define the optional parameters
    parser.add_argument('--verboseOutput', action='store_true', help='Enable verbose output')
    parser.add_argument('--generatePlots', action='store_true', help='Generate plots')
    parser.add_argument('--normalize', action='store_true', help='Normalize data')

    # Define the optional input path parameter '--modelDir'
    parser.add_argument('--modelDir', help='Model directory path')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the parameters
    print("Input path:", args.stackDir)
    print("Verbose output:", args.verboseOutput)
    print("Generate plots:", args.generatePlots)
    print("Normalize data:", args.normalize)
    print("Model directory path:", args.modelDir)

    tdm = TrainDynamicModel(stackDir=args.stackDir,
                            verboseOutput=args.verboseOutput,
                            generatePlots=args.generatePlots,
                            normalize=args.normalize,
                            modelDir=args.modelDir)

    # # ------------------------------------------------------------ TRAIN MODEL
    # print("Train model")
    # start_time = time()
    #
    # # Date and time for output generation
    # dateAndTime = str(datetime.datetime.now())
    # dateAndTimeVector = dateAndTime.split(' ')
    # dateAndTime = dateAndTimeVector[0] + "_" + dateAndTimeVector[1]
    # dateAndTime = dateAndTime.replace(":", "-")
    #
    # # Validation/training ID toggle vectors
    # aliID_validation, aliID_train = utils.generateTrainingValidationVectors(len(normISSAli_train), VALIDATION_SPLIT)
    # misaliID_validation, misaliID_train = utils.generateTrainingValidationVectors(len(normISSMisali_train),
    #                                                                               VALIDATION_SPLIT)
    #
    # # print("aliID_validation: " + str(len(aliID_validation)))
    # # print(sorted(aliID_validation))
    # # print("aliID_train: " + str(len(aliID_train)))
    # # print(sorted(aliID_train))
    # # print("misaliID_validation: " + str(len(misaliID_validation)))
    # # print(sorted(misaliID_validation))
    # # print("misaliID_train: " + str(len(misaliID_train)))
    # # print(sorted(misaliID_train))
    #
    # # Parameters
    # params = {'aliData': normISSAli_train,
    #           'misaliData': normISSMisali_train,
    #           'number_batches': NUMBER_RANDOM_BATCHES,
    #           'batch_size': BATCH_SIZE,
    #           'dim': (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE)}
    #
    # # Generators
    # training_generator = DataGenerator(aliIDs=aliID_train,
    #                                    misaliIDs=misaliID_train,
    #                                    **params)
    # validation_generator = DataGenerator(aliIDs=aliID_validation,
    #                                      misaliIDs=misaliID_validation,
    #                                      **params)
    #
    # # Compile model
    # if not retrainModel:
    #     print("Generating a de novo model for training")
    #     model = compileModel(model=scratchModel(),
    #                          learningRate=LEARNING_RATE)
    # else:
    #     print("Loading pretrained model located at : " + modelDir)
    #     model = load_model(modelDir)
    #
    # # Train model on dataset
    # print("Training model...")
    #
    # dirPath = os.path.join(stackDir, "outputLog_" + dateAndTime)
    #
    # if not os.path.exists(dirPath):
    #     os.makedirs(dirPath)
    #
    # history = model.fit(training_generator,
    #                     validation_data=validation_generator,
    #                     epochs=EPOCHS,
    #                     use_multiprocessing=True,
    #                     workers=1,
    #                     callbacks=getCallbacks(dirPath))
    #
    # myValLoss = np.zeros(1)
    # myValLoss[0] = history.history['val_loss'][-1]
    #
    # np.savetxt(os.path.join(dirPath, "model.txt"), myValLoss)
    # model.save(os.path.join(dirPath, "model.h5"))
    #
    # elapsed_time = time() - start_time
    #
    # print("Time spent training the model: %0.10f seconds." % elapsed_time)
    #
    # if generatePlots:
    #     plotUtils.plotTraining(history, EPOCHS)
    #
    # # ------------------------------------------------------------ TEST MODEL
    # print("\n\nTest model...\n")
    # start_time = time()
    #
    # loadModelDir = os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.h5')
    # model = load_model(loadModelDir)
    #
    # normISS_test, misalignmentInfoVector_test = utils.combineAliAndMisaliVectors(normISSAli_test,
    #                                                                              normISSMisali_test,
    #                                                                              SUBTOMO_SIZE,
    #                                                                              shuffle=False)
    # # Testing the numpy array generation
    # #
    # # import xmippLib as xmipp
    # # inputSubtomoArray1 = xmipp.Image()
    # # inputSubtomo1 = xmipp.Image()
    # #
    # # for i in range(len(normISS_test)):
    # #     X_tmp = normISS_test[i, :]
    # #     inputSubtomoArray1.setData(X_tmp)
    # #     inputSubtomoArray1.write(str(i)+"_test_nparray_subtomo.mrc")
    #
    # # print("len(normISS_test) " + str(len(normISS_test)))
    # # print("len(misalignmentInfoVector_test) " + str(len(misalignmentInfoVector_test)))
    #
    # misalignmentInfoVector_prediction = model.predict(normISS_test)
    #
    # print("misalignmentInfoVector_prediction")
    # print(misalignmentInfoVector_prediction)
    # print("misalignmentInfoVector_test")
    # print(misalignmentInfoVector_test)
    #
    # # Convert the set of probabilities from the previous command into the set of predicted classes
    # misalignmentInfoVector_predictionClasses = np.round(misalignmentInfoVector_prediction)
    #
    # print(misalignmentInfoVector_predictionClasses)
    #
    # np.savetxt(os.path.join(stackDir, 'model_prediction.txt'),
    #            misalignmentInfoVector_predictionClasses)
    #
    # # ------------------------------------------------------------ TRAIN MODEL
    # print("Train model")
    # start_time = time()
    #
    # # Date and time for output generation
    # dateAndTime = str(datetime.datetime.now())
    # dateAndTimeVector = dateAndTime.split(' ')
    # dateAndTime = dateAndTimeVector[0] + "_" + dateAndTimeVector[1]
    # dateAndTime = dateAndTime.replace(":", "-")
    #
    # # Validation/training ID toggle vectors
    # aliID_validation, aliID_train = utils.generateTrainingValidationVectors(len(normISSAli_train), VALIDATION_SPLIT)
    # misaliID_validation, misaliID_train = utils.generateTrainingValidationVectors(len(normISSMisali_train),
    #                                                                               VALIDATION_SPLIT)
    #
    # # print("aliID_validation: " + str(len(aliID_validation)))
    # # print(sorted(aliID_validation))
    # # print("aliID_train: " + str(len(aliID_train)))
    # # print(sorted(aliID_train))
    # # print("misaliID_validation: " + str(len(misaliID_validation)))
    # # print(sorted(misaliID_validation))
    # # print("misaliID_train: " + str(len(misaliID_train)))
    # # print(sorted(misaliID_train))
    #
    # # Parameters
    # params = {'aliData': normISSAli_train,
    #           'misaliData': normISSMisali_train,
    #           'number_batches': NUMBER_RANDOM_BATCHES,
    #           'batch_size': BATCH_SIZE,
    #           'dim': (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE)}
    #
    # # Generators
    # training_generator = DataGenerator(aliIDs=aliID_train,
    #                                    misaliIDs=misaliID_train,
    #                                    **params)
    # validation_generator = DataGenerator(aliIDs=aliID_validation,
    #                                      misaliIDs=misaliID_validation,
    #                                      **params)
    #
    # # Compile model
    # if not retrainModel:
    #     print("Generating a de novo model for training")
    #     model = compileModel(model=scratchModel(),
    #                          learningRate=LEARNING_RATE)
    # else:
    #     print("Loading pretrained model located at : " + modelDir)
    #     model = load_model(modelDir)
    #
    # # Train model on dataset
    # print("Training model...")
    #
    # dirPath = os.path.join(stackDir, "outputLog_" + dateAndTime)
    #
    # if not os.path.exists(dirPath):
    #     os.makedirs(dirPath)
    #
    # history = model.fit(training_generator,
    #                     validation_data=validation_generator,
    #                     epochs=EPOCHS,
    #                     use_multiprocessing=True,
    #                     workers=1,
    #                     callbacks=getCallbacks(dirPath))
    #
    # myValLoss = np.zeros(1)
    # myValLoss[0] = history.history['val_loss'][-1]
    #
    # np.savetxt(os.path.join(dirPath, "model.txt"), myValLoss)
    # model.save(os.path.join(dirPath, "model.h5"))
    #
    # elapsed_time = time() - start_time
    #
    # print("Time spent training the model: %0.10f seconds." % elapsed_time)
    #
    # if generatePlots:
    #     plotUtils.plotTraining(history, EPOCHS)
    #
    # # ------------------------------------------------------------ TEST MODEL
    # print("\n\nTest model...\n")
    # start_time = time()
    #
    # loadModelDir = os.path.join(stackDir, "outputLog_" + dateAndTime + '/model.h5')
    # model = load_model(loadModelDir)
    #
    # normISS_test, misalignmentInfoVector_test = utils.combineAliAndMisaliVectors(normISSAli_test,
    #                                                                              normISSMisali_test,
    #                                                                              SUBTOMO_SIZE,
    #                                                                              shuffle=False)
    # # Testing the numpy array generation
    # #
    # # import xmippLib as xmipp
    # # inputSubtomoArray1 = xmipp.Image()
    # # inputSubtomo1 = xmipp.Image()
    # #
    # # for i in range(len(normISS_test)):
    # #     X_tmp = normISS_test[i, :]
    # #     inputSubtomoArray1.setData(X_tmp)
    # #     inputSubtomoArray1.write(str(i)+"_test_nparray_subtomo.mrc")
    #
    # # print("len(normISS_test) " + str(len(normISS_test)))
    # # print("len(misalignmentInfoVector_test) " + str(len(misalignmentInfoVector_test)))
    #
    # misalignmentInfoVector_prediction = model.predict(normISS_test)
    #
    # print("misalignmentInfoVector_prediction")
    # print(misalignmentInfoVector_prediction)
    # print("misalignmentInfoVector_test")
    # print(misalignmentInfoVector_test)
    #
    # # Convert the set of probabilities from the previous command into the set of predicted classes
    # misalignmentInfoVector_predictionClasses = np.round(misalignmentInfoVector_prediction)
    #
    # print(misalignmentInfoVector_predictionClasses)
    #
    # np.savetxt(os.path.join(stackDir, 'model_prediction.txt'),
    #            misalignmentInfoVector_predictionClasses)
    #
    # elapsed_time = time() - start_time
    # print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)
    #
    # mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
    # print("Final model mean absolute error val_loss: %f\n" % mae)
    #
    # loss = model.evaluate(normISS_test,
    #                       misalignmentInfoVector_test,
    #                       verbose=2)
    #
    # print("Testing set mean absolute error: {:5.4f}".format(loss[0]))
    # print("Testing set accuracy: {:5.4f}".format(loss[1]))
    #
    # # Plot results from testing
    # if generatePlots:
    #     plotUtils.plotTesting(
    #         misalignmentInfoVector_test,
    #         misalignmentInfoVector_predictionClasses
    #     )
    # elapsed_time = time() - start_time
    # print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)
    #
    # mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
    # print("Final model mean absolute error val_loss: %f\n" % mae)
    #
    # loss = model.evaluate(normISS_test,
    #                       misalignmentInfoVector_test,
    #                       verbose=2)
    #
    # print("Testing set mean absolute error: {:5.4f}".format(loss[0]))
    # print("Testing set accuracy: {:5.4f}".format(loss[1]))
    #
    # # Plot results from testing
    # if generatePlots:
    #     plotUtils.plotTesting(
    #         misalignmentInfoVector_test,
    #         misalignmentInfoVector_predictionClasses
    #     )
