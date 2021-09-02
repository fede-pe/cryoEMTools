
""" Module to save the obtained subtomos and its misalignment information to posteriorly train and test the network.
Also, it generates the output numpy vectors to input the network. """

import errno
import os
import sys
import csv
import glob


def addSubtomosToOutput(pathPatternToSubtomoFiles, alignmentFlag):
    """ This methods add to the output metadata file (or creates it if it does not exist) the imported subtomos form
    the regex, indicating if the are or not aligned. """

    fieldNames = ['subTomoPath', 'alignmentToggle']

    fileName = 'misalignmentMetadata.txt'
    filePrefix = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'trainingSet')
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


# ----------------------------------- Main ------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] != "0" or sys.argv[2] != "1" or sys.argv[3] != "0" or sys.argv[3] != "1":
        print("Usage: python generateTrainingSet.py <pathPatternToSubtomoFiles> <alignmentFlag 1/0> "
              "<generateNetworkInputFlag 1/0>\n"
              "<pathPatternToSubtomoFiles>: Path to the folder containing the subtomo volume files.\n"
              "<alignmentFlag 1/0>: Flag to set the imported subtomos as aligned (1) or misaligned (0). \n"
              "<generateNetworkInputFlag 1/0>: Generate the output numpy vectors to input the network for training and "
              "testing.\n")
        exit()

    pathPatternToSubtomoFiles = sys.argv[1]
    alignmentFlag = sys.argv[2]
    generateNetworkInputFlag = sys.argv[3]

