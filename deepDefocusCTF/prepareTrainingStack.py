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

    # df_metadata_2 = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
    # print('read the dataframe')
    # print(df_metadata_2)

    for line in metadataLines:
        #storedFile = line[39:]  #[88:]
        storedFile = int(line[88:])
        #subset = int(line[30:38]) #esto esta "mal" deberia ser [30:39]  =~  [76:87]
        subset = int(line[76:87])
        #defocus = int(line[10:21]) #no estoy seguro que lo este cogiendo bien salvo que 9.7d == 10     =~ [10:21] [21:32]
        defocus_U = int(line[10:21])
        defocus_V = int(line[21:32])
        dSinA = int(line[32:43])
        dCosA = int(line[43:54])

        img1Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_1.xmp")
        img2Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_2.xmp")
        img3Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_3.xmp")
        img1 = xmipp.Image(img1Path).getData()
        img2 = xmipp.Image(img2Path).getData()
        img3 = xmipp.Image(img3Path).getData()
        imagMatrix[i, :, :, 0] = img1
        imagMatrix[i, :, :, 1] = img2
        imagMatrix[i, :, :, 2] = img3

        #defocusVector.append(defocus)
        defocusVector.append((defocus_U, defocus_V, dSinA, dCosA))

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
