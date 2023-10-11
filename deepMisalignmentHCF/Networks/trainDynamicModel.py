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
EPOCHS = 50  # Number of epochs
LEARNING_RATE = 0.0001  # Learning rate
TESTING_SPLIT = 0.1  # Ratio of data used for testing
VALIDATION_SPLIT = 0.15  # Ratio of data used for validation


class TrainDynamicModel:
    """ This class holds all the methods needed to train dynamically a DNN model """

    def __init__(self, stackDir, verboseOutput, generatePlots, normalize, modelDir, debug):
        self.stackDir = stackDir
        self.verboseOutput = verboseOutput
        self.generatePlots = generatePlots
        self.normalize = normalize
        self.modelDir = modelDir
        self.debug = debug

        if modelDir is None:
            self.retrainModel = False
        else:
            self.retrainModel = True

        # Side information variables
        self.totalSubtomos = 0
        self.totalAliSubtomos = 0
        self.totalMisaliSubtomos = 0
        self.numberRandomBatches = -1

        # Dictionaries with dataset information
        self.aliDict = {}
        self.misaliDict = {}
        self.aliDict_test = {}
        self.misaliDict_test = {}

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
        self.splitDataTrainTest()
        self.splitDataTrainValidation()

        self.modelTraining()
        self.modelTesting()

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

        # Merge datasets in case any of them do not reach the minimum size
        minPercentage = min(VALIDATION_SPLIT, TESTING_SPLIT)
        minSizeAli = BATCH_SIZE // (2 * len(self.aliDict.keys()))
        minSizeMisali = BATCH_SIZE // (2 * len(self.misaliDict.keys()))

        if self.debug:
            print("Merging datasets...")
            print("Minimum size for aligned datasets %d" % minSizeAli)
            print("Minimum size for misaligned datasets %d" % minSizeMisali)

        aliDatasetsToMerge = []
        aliKeysToRemove = []

        for key in self.aliDict.keys():
            if (self.aliDict[key][1] * minPercentage) / 4 < minSizeAli:
                aliDatasetsToMerge.append(self.aliDict[key][0])
                aliKeysToRemove.append(key)

        if len(aliDatasetsToMerge) == 1:
            for key in self.misaliDict.keys():
                if key != aliKeysToRemove[0]:
                    aliDatasetsToMerge.append(self.misaliDict[key][0])
                    aliKeysToRemove.append(key)
                    break

        for key in aliKeysToRemove:
            del self.aliDict[key]

        if len(aliDatasetsToMerge) > 0:
            mergedAliDataset = np.concatenate(aliDatasetsToMerge)
            self.aliDict["mergedAliDataset"] = (mergedAliDataset, len(mergedAliDataset))

        misaliDatasetToMerge = []
        misaliKeysToRemove = []

        for key in self.misaliDict.keys():
            if (self.misaliDict[key][1] * minPercentage) / 4 < minSizeMisali:
                misaliDatasetToMerge.append(self.misaliDict[key][0])
                misaliKeysToRemove.append(key)

        if len(misaliDatasetToMerge) == 1:
            for key in self.misaliDict.keys():
                if key != misaliKeysToRemove[0]:
                    misaliDatasetToMerge.append(self.misaliDict[key][0])
                    misaliKeysToRemove.append(key)
                    break

        for key in misaliKeysToRemove:
            del self.misaliDict[key]

        if len(misaliDatasetToMerge) > 0:
            mergedMisaliDataset = np.concatenate(misaliDatasetToMerge)
            self.misaliDict["mergeMisaliDataset"] = (mergedMisaliDataset, len(mergedMisaliDataset))

        # Update the number of random batches respect to the dataset and batch sizes
        self.numberRandomBatches = self.totalSubtomos // BATCH_SIZE

        # Output classes distribution info
        aliSubtomosRatio = self.totalAliSubtomos / self.totalSubtomos
        misaliSubtomosRatio = self.totalMisaliSubtomos / self.totalSubtomos

        if self.verboseOutput:
            print("\nClasses distribution:")

            print("\tAligned datasets:")
            for key in self.aliDict.keys():
                if self.verboseOutput:
                    print("\t\tDataset %s:\t"
                          "Aligned: %d\t"
                          % (key,
                             self.aliDict[key][1]))

            print("\tMisaligned datasets:")
            for key in self.misaliDict.keys():
                if self.verboseOutput:
                    print("\t\tDataset %s:\t"
                          "Misaligned: %d\t"
                          % (key,
                             self.misaliDict[key][1]))

            if self.verboseOutput:
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

        print("Aligned datasets:")
        for key in self.aliDict.keys():
            # Normalize input subtomo data stream to N(0,1)
            self.aliDict[key] = (utils.normalizeInputDataStream(self.aliDict[key][0]), self.aliDict[key][1])

            # Test normalization
            if self.debug:
                print("\tDataset %s:\t"
                      "Aligned: mean %s, std %s\t"
                      % (key,
                         str(np.mean(self.aliDict[key][0])), str(np.std(self.aliDict[key][0]))))

        print("Misaligned datasets:")
        for key in self.misaliDict.keys():
            # Normalize input subtomo data stream to N(0,1)
            self.misaliDict[key] = (utils.normalizeInputDataStream(self.misaliDict[key][0]), self.misaliDict[key][1])

            # Test normalization
            if self.debug:
                print("\tDataset %s:\t"
                      "Misaligned: mean %s, std %s\t"
                      % (key,
                         str(np.mean(self.misaliDict[key][0])), str(np.std(self.misaliDict[key][0]))))

        norm_time = time() - start_time
        print("Time spent in data normalization: %0.10f seconds.\n" % norm_time)

    def dataAugmentation(self):
        """ Method to perform data augmentation strategies to input data """

        print("------------------------------------------ Data augmentation")
        start_time = time()

        # Data augmentation for aligned subtomos
        print("Data augmentation for aligned datasets")

        for key in self.aliDict.keys():
            if self.debug:
                print("\tData augmentation for dataset: %s" % key)

            numberOfAliSubtomos = self.aliDict[key][1]
            inputSubtomoStreamAli = self.aliDict[key][0]

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

            if self.debug:
                print("\tData structure shapes for aligned dataset BEFORE augmentation:" +
                      str(inputSubtomoStreamAli.shape))

            inputSubtomoStreamAli = np.concatenate((inputSubtomoStreamAli,
                                                    generatedSubtomosArrayAli))

            self.aliDict[key] = (inputSubtomoStreamAli, numberOfAliSubtomos)

            if self.debug:
                print("\tData structure shapes for aligned dataset AFTER augmentation:" +
                      str(inputSubtomoStreamAli.shape))
                print("\tNumber of new subtomos generated: %d" % len(generatedSubtomosAli))

        # Data augmentation for misaligned subtomos
        print("Data augmentation for misaligned datasets")

        for key in self.misaliDict.keys():
            if self.debug:
                print("\tData augmentation for dataset: %s" % key)

            numberOfMisaliSubtomos = self.misaliDict[key][1]
            inputSubtomoStreamMisali = self.misaliDict[key][0]

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

            if self.debug:
                print("\tData structure shapes for misaligned dataset BEFORE augmentation:" +
                      str(inputSubtomoStreamMisali.shape))

            inputSubtomoStreamMisali = np.concatenate((inputSubtomoStreamMisali,
                                                       generatedSubtomosArrayMisali))

            self.misaliDict[key] = (inputSubtomoStreamMisali, numberOfMisaliSubtomos)

            if self.debug:
                print("\tData structure shapes for misaligned dataset AFTER augmentation:" +
                      str(inputSubtomoStreamMisali.shape))
                print("\tNumber of new subtomos generated: %d" % len(generatedSubtomosMisali))

        dataAug_time = time() - start_time
        print("Time spent in data augmentation: %0.10f seconds.\n\n" % dataAug_time)

    def splitDataTrainTest(self):
        """ Method to split data into train and test"""

        print("------------------------------------------ Data split train-test")
        start_time = time()

        # Aligned subtomos
        if self.debug:
            print('Data split for aligned datasets')

        for key in self.aliDict.keys():
            if self.debug:
                print("\tSplit data for dataset: %s" % key)

            aliTrain, aliTest = train_test_split(self.aliDict[key][0],
                                                 test_size=TESTING_SPLIT,
                                                 random_state=42)

            self.aliDict[key] = (aliTrain, len(aliTrain))
            self.aliDict_test[key] = (aliTest, len(aliTest))

            if self.debug:
                print('\t\tData objects final dimensions for dataset %s' % key)
                print('\t\tTrain aligned subtomos matrix: ' + str(np.shape(aliTrain)))
                print('\t\tTest aligned subtomos matrix: ' + str(np.shape(aliTest)))

        # Misaligned subtomos
        if self.debug:
            print('Data split for misaligned dataset')

        for key in self.misaliDict.keys():
            if self.debug:
                print("\tSplit data for dataset: %s" % key)

            misaliTrain, misaliTest = train_test_split(self.misaliDict[key][0],
                                                       test_size=TESTING_SPLIT,
                                                       random_state=42)

            self.misaliDict[key] = (misaliTrain, len(misaliTrain))
            self.misaliDict_test[key] = (misaliTest, len(misaliTest))

            if self.debug:
                print('\t\tData objects final dimensions for dataset %s' % key)
                print('\t\tTrain misaligned subtomos matrix: ' + str(np.shape(misaliTrain)))
                print('\t\tTest misaligned subtomos matrix: ' + str(np.shape(misaliTest)))

        elapsed_time = time() - start_time
        print("Time spent in data train-test splitting: %0.10f seconds.\n\n" % elapsed_time)

    def splitDataTrainValidation(self):
        """ Method to split data into train and validation"""

        print("------------------------------------------ Data split train-validation")
        start_time = time()

        # Split validation/training for aligned data
        if self.debug:
            print('Data split for aligned datasets')

        for key in self.aliDict.keys():
            if self.debug:
                print("\tSplit data for dataset: %s" % key)

            # Validation/training ID toggle vectors
            aliID_validation, aliID_train = utils.generateTrainingValidationVectors(self.aliDict[key][1],
                                                                                    VALIDATION_SPLIT)

            # Add ID's for training and validation to dictionary
            self.aliDict[key] = (self.aliDict[key][0],
                                 self.aliDict[key][1],
                                 aliID_train,
                                 aliID_validation)

            if self.debug:
                print("\t\tShape of the full set before split %s: " % str(np.shape(self.aliDict[key][0])))
                print("\t\tSize of train split: %d " % len(aliID_train))
                print("\t\tSize of validation split: %d " % len(aliID_validation))

        # Split validation/training for misaligned data
        if self.debug:
            print('Data split for misaligned datasets')

        for key in self.misaliDict.keys():
            if self.debug:
                print("\tSplit data for dataset: %s" % key)

            # Validation/training ID toggle vectors
            misaliID_validation, misaliID_train = utils.generateTrainingValidationVectors(self.misaliDict[key][1],
                                                                                          VALIDATION_SPLIT)

            # Add ID's for training and validation to dictionary
            self.misaliDict[key] = (self.misaliDict[key][0],
                                    self.misaliDict[key][1],
                                    misaliID_train,
                                    misaliID_validation)

            if self.debug:
                print("\t\tShape of the full set before split %s: " % str(np.shape(self.misaliDict[key][0])))
                print("\t\tSize of train split: %d " % len(misaliID_train))
                print("\t\tSize of validation split: %d " % len(misaliID_validation))

        elapsed_time = time() - start_time
        print("Time spent in data train-test splitting: %0.10f seconds.\n\n" % elapsed_time)

    def modelTraining(self):
        """ Method for model training """

        print("------------------------------------------ Model training")
        start_time = time()

        # Parameters
        params = {'aliDict': self.aliDict,
                  'misaliDict': self.misaliDict,
                  'number_batches': self.numberRandomBatches,
                  'batch_size': BATCH_SIZE,
                  'dim': (SUBTOMO_SIZE, SUBTOMO_SIZE, SUBTOMO_SIZE)}

        # Generators
        training_generator = DataGenerator(mode=0, **params)
        validation_generator = DataGenerator(mode=1, **params)

        # Compile model
        if not self.retrainModel:
            print("Generating a de novo model for training")
            model = compileModel(model=scratchModel(),
                                 learningRate=LEARNING_RATE)
        else:
            print("Loading pretrained model located at : " + self.modelDir)
            model = load_model(self.modelDir)

        # Train model on dataset
        print("Training model...")

        history = model.fit(training_generator,
                            validation_data=validation_generator,
                            epochs=EPOCHS,
                            use_multiprocessing=True,
                            workers=1,
                            callbacks=getCallbacks(self.dirPath))

        myValLoss = np.zeros(1)
        myValLoss[0] = history.history['val_loss'][-1]

        np.savetxt(os.path.join(self.dirPath, "model.txt"), myValLoss)
        model.save(os.path.join(self.dirPath, "model.h5"))

        if self.generatePlots:
            plotUtils.plotTraining(history, EPOCHS, self.dirPath)

        elapsed_time = time() - start_time
        print("Time spent training the model: %0.10f seconds.\n\n" % elapsed_time)

    def modelTesting(self):
        """ Method for model testing"""

        print("------------------------------------------ Model testing")
        start_time = time()

        loadModelDir = os.path.join(self.dirPath, "model.h5")
        model = load_model(loadModelDir)

        normISS_test, misalignmentInfoVector_test = utils.combineAliAndMisaliVectors(self.aliDict_test,
                                                                                     self.misaliDict_test,
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

        if self.debug:
            print("misalignmentInfoVector_prediction")
            print(misalignmentInfoVector_prediction)
            print("misalignmentInfoVector_test")
            print(misalignmentInfoVector_test)

        # Convert the set of probabilities from the previous command into the set of predicted classes
        misalignmentInfoVector_predictionClasses = np.round(misalignmentInfoVector_prediction)

        if self.debug:
            print(misalignmentInfoVector_predictionClasses)

        np.savetxt(os.path.join(self.stackDir, 'model_prediction.txt'),
                   misalignmentInfoVector_predictionClasses)

        mae = mean_absolute_error(misalignmentInfoVector_test, misalignmentInfoVector_predictionClasses)
        print("Final model mean absolute error val_loss: %f\n" % mae)

        loss = model.evaluate(normISS_test,
                              misalignmentInfoVector_test,
                              verbose=2)

        print("Testing set mean absolute error: {:5.4f}".format(loss[0]))
        print("Testing set accuracy: {:5.4f}".format(loss[1]))

        # Plot results from testing
        if self.generatePlots:
            plotUtils.plotTesting(
                misalignmentInfoVector_test,
                misalignmentInfoVector_predictionClasses,
                self.dirPath
            )

        elapsed_time = time() - start_time
        print("Time spent testing the model: %0.10f seconds.\n" % elapsed_time)


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
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

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
    print("Debug mode:", args.debug)

    tdm = TrainDynamicModel(stackDir=args.stackDir,
                            verboseOutput=args.verboseOutput,
                            generatePlots=args.generatePlots,
                            normalize=args.normalize,
                            modelDir=args.modelDir,
                            debug=args.debug)
