import numpy as np
import os
import csv
import sys
import xmippLib as xmipp
from time import time


def prepareData(stackDir):
    with open(os.path.join(stackDir, "metadata.txt")) as f:
        metadataLines = csv.DictReader(f)

    # fieldNames = ['maxDistance', 'totalDistance', 'hullArea', 'hullPerimeter', 'pcaX', 'pcaY', 'subTomoPath']

    Ndim = len(list(metadataLines))
    inputDataStream = np.zeros((Ndim, 32, 32, 32), dtype=np.float64)
    misalignmentInfoList = []

    for i, line in enumerate(metadataLines):
        if i == 0:
            pass

        misalignmentInfoVector = [line["maxDistance"],
                                  line["totalDistance"],
                                  line["hullArea"],
                                  line["hullPerimeter"],
                                  line["pcaX"],
                                  line["pcaY"]]

        subtomoPath = line["subTomoPath"]

        subtomoVol = xmipp.Image(subtomoPath).getData()

        inputDataStream[i, :, :, :] = subtomoVol

        misalignmentInfoList.append(misalignmentInfoVector)

    inputDataStreamPath = os.path.join(stackDir, "inputDataStream.npy")
    misalignmentInfoPath = os.path.join(stackDir, "misalignmentInfoList.npy")

    np.save(inputDataStreamPath, inputDataStream)
    np.save(misalignmentInfoPath, misalignmentInfoList)


if __name__ == "__main__":
    stackDir = sys.argv[1]

    print("Preparing stack...")
    start_time = time()
    prepareData(stackDir)
    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)
