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


# ----------------------------------- Main ------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Preparing stack...")
        start_time = time()

        generateNetworkVectors()

        elapsed_time = time() - start_time
        print("Time spent preparing the data: %0.10f seconds." % elapsed_time)

    elif len(sys.argv) == 3 and (sys.argv[3] != "0" or sys.argv[3] != "1"):
        pathPatternToSubtomoFiles = sys.argv[1]
        alignmentFlag = sys.argv[2]

        addSubtomosToOutput(pathPatternToSubtomoFiles, alignmentFlag)

    else:
        print("2 options usage:\n\n"
              ""
              "Option 1: Add subtomos to the dataset indicating if they are aligned or not:\n"
              "python generateDataset.py <pathPatternToSubtomoFiles> <alignmentFlag 1/0> \n"
              "<pathPatternToSubtomoFiles>: Regex path to the subtomo volume files.\n"
              "<alignmentFlag 1/0>: Flag to set the imported subtomos as aligned (1) or misaligned (0). \n\n"
              ""
              "Option 2: If no options are entered the program will create the output vectors for posteriorly train "
              "and test the network:\n"
              "python generateDataset.py")
