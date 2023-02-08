import numpy as np
import os
import sys
import xmippLib as xmipp
import pandas as pd
from time import time


def prepareData(stackDir):
    df_metadata = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
    Ndim = df_metadata.shape[0]
    imagMatrix = np.zeros((Ndim, 512, 512, 3), dtype=np.float64)
    defocusVector = np.zeros((Ndim, 4), dtype=np.float64)
    i = 0

    for index in df_metadata.index.to_list():
        storedFile = df_metadata.loc[index, 'FILE']
        subset = df_metadata.loc[index, 'SUBSET']
        defocus_U = df_metadata.loc[index, 'DEFOCUS_U']
        defocus_V = df_metadata.loc[index, 'DEFOCUS_V']
        dSinA = df_metadata.loc[index, 'Sin(2*angle)']
        dCosA = df_metadata.loc[index, 'Cos(2*angle)']

        # Replace is done since we want the 3 images not only the one in the metadata file
        img1Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_1.xmp")
        img2Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_2.xmp")
        img3Path = storedFile.replace("_psdAt_%d.xmp" % subset, "_psdAt_3.xmp")

        img1 = xmipp.Image(img1Path).getData()
        img2 = xmipp.Image(img2Path).getData()
        img3 = xmipp.Image(img3Path).getData()

        imagMatrix[i, :, :, 0] = img1
        imagMatrix[i, :, :, 1] = img2
        imagMatrix[i, :, :, 2] = img3

        defocusVector[i, 0] = int(defocus_U)
        defocusVector[i, 1] = int(defocus_V)
        defocusVector[i, 2] = dSinA
        defocusVector[i, 3] = dCosA

        i += 1

    imageStackDir = os.path.join(stackDir, "preparedImageStack.npy")
    defocusStackDir = os.path.join(stackDir, "preparedDefocusStack.npy")

    np.save(imageStackDir, imagMatrix)
    np.save(defocusStackDir, defocusVector)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 prepareTrainingStack_deprecated.py <dirOut>")
        exit(0)

    stackDir = sys.argv[1]
    print("Preparing stack...")
    start_time = time()
    prepareData(stackDir)
    elapsed_time = time() - start_time
    print("Time spent preparing the data: %0.10f seconds." % elapsed_time)
    exit(0)
