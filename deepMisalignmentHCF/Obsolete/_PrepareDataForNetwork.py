import numpy as np
import os
import csv
import sys
import xmippLib as xmipp
from time import time


def prepareData(stackDir):
    with open(os.path.join(stackDir, "misalignmentStatistics.txt")) as f:
        metadataLines = csv.DictReader(f, delimiter='\t')

        # fieldNames = ['maxDistance', 'totalDistance', 'hullArea', 'hullPerimeter', 'pcaX', 'pcaY', 'subTomoPath']

        Ndim = 0
        misalignmentInfoList = []
        subtomoPathList = []

        # Complete misalignmentInfoList vector.
        for i, line in enumerate(metadataLines):
            misalignmentInfoVector = [float(line["centroidX"]),
                                      float(line["centroidY"]),
                                      float(line["maxDistance"]),
                                      float(line["totalDistance"]),
                                      float(line["hullArea"]),
                                      float(line["hullPerimeter"]),
                                      float(line["pcaX"]),
                                      float(line["pcaY"])]

            subtomoPathList.append(line["subTomoPath"])
            misalignmentInfoList.append(misalignmentInfoVector)

            Ndim += 1

        inputDataStream = np.zeros((Ndim, 32, 32, 32), dtype=np.float64)

        # Complete inputDataStream matrix (we only can iterate over the csvReader once and it is necessary to know the
        # Ndim a priori.
        for i, subtomoPath in enumerate(subtomoPathList):
            subtomoVol = xmipp.Image(subtomoPath).getData()
            inputDataStream[i, :, :, :] = subtomoVol

        inputDataStreamPath = os.path.join(stackDir, "inputDataStream.npy")
        misalignmentInfoPath = os.path.join(stackDir, "misalignmentInfoList.npy")

        np.save(inputDataStreamPath, inputDataStream)
        np.save(misalignmentInfoPath, misalignmentInfoList)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python <pathStackDir> \n"
              "<pathStackDir>: Path to folder containing the subtomo volumes and the misalignment information file ")
        exit()

    stackDir = sys.argv[1]

    print("Preparing stack...")
    start_time = time()
    prepareData(stackDir)
    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)
