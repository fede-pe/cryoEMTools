import numpy as np
import os
import sys
import xmippLib as xmipp
from time import time


def prepareData(stackDir):
    metadataFile = open(os.path.join(stackDir, "metadata.txt"))
    metadataLines = metadataFile.read().splitlines()
    metadataLines.pop(0)
    Ndim = len(metadataLines)
    imagMatrix = np.zeros((Ndim, 512, 512, 3), dtype=np.float64)
    defocusVector = []
    i = 0
    for line in metadataLines:
        storedFile = line[39:]
        subset = int(line[30:38])
        defocus = int(line[10:21])
        img1Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_1.xmp")
        img2Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_2.xmp")
        img3Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_3.xmp")
        img1 = xmipp.Image(img1Path).getData()
        img2 = xmipp.Image(img2Path).getData()
        img3 = xmipp.Image(img3Path).getData()
        imagMatrix[i, :, :, 0] = img1
        imagMatrix[i, :, :, 1] = img2
        imagMatrix[i, :, :, 2] = img3
        defocusVector.append(defocus)
        i += 1
    imageStackDir = os.path.join(stackDir, "preparedImageStack.npy")
    defocusStackDir = os.path.join(stackDir, "preparedDefocusStack.npy")
    np.save(imageStackDir, imagMatrix)
    np.save(defocusStackDir, defocusVector)


if __name__ == "__main__":
    stackDir = sys.argv[1]
    print("Preparing stack...")
    start_time = time()
    prepareData(stackDir)
    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)
