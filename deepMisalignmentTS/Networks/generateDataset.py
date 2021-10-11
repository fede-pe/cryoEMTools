""" Module to save the obtained subtomos and its misalignment information to posteriorly train and test the network.
Also, it generates the output numpy vectors to input the network. """

import errno
import os
import sys
import csv
import glob

import numpy as np
import xmippLib as xmipp
from time import time


def addSubtomosToOutput(pathPatternToSubtomoFiles, alignmentFlag):
    """ This methods add to the output metadata file (or creates it if it does not exist) the imported subtomos form
    the regex, indicating if the are or not aligned. """

    fieldNames = ['subTomoPath', 'alignmentToggle']

    fileName = 'misalignmentMetadata.txt'
    filePrefix = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '../trainingSet')
    filePath = os.path.join(filePrefix, fileName)

    # Search for the last subtomo saved
    lastIndex = 0
    while True:
        if os.path.exists(os.path.join(filePrefix, "subtomo%s.mrc" % str(lastIndex).zfill(8))):
            lastIndex += 1
        else:
            break

    for file in glob.glob(pathPatternToSubtomoFiles):
        " Create intermediate directories if missing "
        if not os.path.exists(filePath):
            try:
                os.makedirs(os.path.dirname(filePath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        mode = "a" if os.path.exists(filePath) else "w"

        with open(filePath, mode) as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldNames)

            if mode == "w":
                writer.writeheader()

            writerDict = {
                'subTomoPath': file,
                'alignmentToggle': alignmentFlag,
            }

            writer.writerow(writerDict)

        os.symlink(file, os.path.join(filePrefix, "subtomo%s.mrc" % str(lastIndex).zfill(8)))
        lastIndex += 1

    print("Added %d subtomos to dataset." % len(glob.glob(pathPatternToSubtomoFiles)))


def generateNetworkVectors():
    """ This method generates the vectors associated to the metadata files for posteriorly training and testing the
    network. """

    fileName = 'misalignmentMetadata.txt'
    filePrefix = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '../trainingSet')
    filePath = os.path.join(filePrefix, fileName)

    with open(filePath) as f:
        metadataLines = csv.DictReader(f, delimiter='\t')

        # fieldNames = ['subTomoPath', 'alignmentToggle']

        Ndim = 0
        misalignmentInfoList = []
        subtomoPathList = []

        # Complete misalignmentInfoList vector.
        for i, line in enumerate(metadataLines):
            misalignmentInfoList.append(int(line["alignmentToggle"]))

            subtomoPathList.append(line["subTomoPath"])

            Ndim += 1

        inputDataStream = np.zeros((Ndim, 32, 32, 32), dtype=np.float64)

        # Complete inputDataStream matrix (it is only possible to iterate over the csvReader once and it is necessary
        # to know the Ndim a priori.
        for i, subtomoPath in enumerate(subtomoPathList):
            subtomoVol = xmipp.Image(subtomoPath).getData()
            inputDataStream[i, :, :, :] = subtomoVol

        inputDataStreamPath = os.path.join(filePrefix, "inputDataStream.npy")
        misalignmentInfoPath = os.path.join(filePrefix, "misalignmentInfoList.npy")

        np.save(inputDataStreamPath, inputDataStream)
        np.save(misalignmentInfoPath, misalignmentInfoList)

        print("Output subtomo vector saved at" + inputDataStreamPath)
        print("Output misalignment info vector saved at" + misalignmentInfoPath)


def generateNetworkVectorsSplit():
    """ This method generates the vectors associated to the metadata files for posteriorly training and testing the
    network. """

    fileName = 'misalignmentMetadata.txt'
    filePrefix = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '../trainingSet')
    filePath = os.path.join(filePrefix, fileName)

    with open(filePath) as f:
        metadataLines = csv.DictReader(f, delimiter='\t')

        # fieldNames = ['subTomoPath', 'alignmentToggle']

        NdimAli = 0
        NdimMisali = 0
        subtomoPathListAli = []
        subtomoPathListMisali = []

        # Split subtomo paths between aligned and misaligned sets
        for i, line in enumerate(metadataLines):
            # Aligned
            if int(line["alignmentToggle"]) == 1:
                subtomoPathListAli.append(line["subTomoPath"])
                NdimAli += 1

            # Misaligned
            elif int(line["alignmentToggle"]) == 0:
                subtomoPathListMisali.append(line["subTomoPath"])
                NdimMisali += 1

        inputDataStreamAli = np.zeros((NdimAli, 32, 32, 32), dtype=np.float64)
        inputDataStreamMisali = np.zeros((NdimMisali, 32, 32, 32), dtype=np.float64)

        # Complete inputDataStreamAli and inputDataStreamMisali matrix (it is only possible to iterate over the
        # csvReader once and it is necessary to know NdimAli and Misali a priori.
        for i, subtomoPath in enumerate(subtomoPathListAli):
            subtomoVol = xmipp.Image(subtomoPath).getData()
            inputDataStreamAli[i, :, :, :] = subtomoVol

        for i, subtomoPath in enumerate(subtomoPathListMisali):
            subtomoVol = xmipp.Image(subtomoPath).getData()
            inputDataStreamMisali[i, :, :, :] = subtomoVol

        inputDataStreamAliPath = os.path.join(filePrefix, "inputDataStreamAli.npy")
        inputDataStreamMisaliPath = os.path.join(filePrefix, "inputDataStreamMisali.npy")

        np.save(inputDataStreamAliPath, inputDataStreamAli)
        np.save(inputDataStreamMisaliPath, inputDataStreamMisali)

        print("Output aligned subtomo vector saved at" + inputDataStreamAliPath)
        print("Output misaligned subtomo vector saved at" + inputDataStreamMisaliPath)


# ----------------------------------- Main ------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] != "0" or sys.argv[1] != "1"):
        start_time = time()

        splitFlag = int(sys.argv[1])

        if splitFlag == 0:
            print("Preparing stack...")
            generateNetworkVectors()
        elif splitFlag == 1:
            print("Preparing stack in split mode...")
            generateNetworkVectorsSplit()

        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

    elif len(sys.argv) == 3 and (sys.argv[2] != "0" or sys.argv[2] != "1"):
        print("Adding subtomos to dataset...")
        start_time = time()

        pathPatternToSubtomoFiles = sys.argv[1]
        alignmentFlag = sys.argv[2]

        addSubtomosToOutput(pathPatternToSubtomoFiles, alignmentFlag)

        elapsed_time = time() - start_time
        print("Time spent collecting data: %0.10f seconds." % elapsed_time)

    else:
        print("2 options usage:\n\n"
              ""
              "Option 1: Add subtomos to the dataset indicating if they are aligned or not:\n"
              "python generateDataset.py <pathPatternToSubtomoFiles> <alignmentFlag 1/0> \n"
              "<pathPatternToSubtomoFiles>: Regex path to the subtomo volume files.\n"
              "<alignmentFlag 1/0>: Flag to set the imported subtomos as aligned (1) or misaligned (0). \n\n"
              ""
              "Option 2: Create the output vectors indicating if the misaligned and aligned subtomos are split in two "
              "different vectors (1) or not (0). if not, a second vector containing the misalignment information would "
              "be generated:\n"
              "python generateDataset.py <splitVectorFlag 1/0>\n"
              "<splitVectorFlag 1/0>: Flag to set if the output vectors must be split.")
