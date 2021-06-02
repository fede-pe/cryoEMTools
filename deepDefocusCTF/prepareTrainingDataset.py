import os
import sys

import numpy as np
import re
import xmippLib as xmipp
import shutil
import sqlite3
import pandas as pd

class DeepDefocus:

    @staticmethod
    def importCTF(fnDir, dataFlag):
        fileList = []
        verbose = 0
        #CONNECT TO THE DATABASE
        dbName = 'subset.sqlite'
        dbRoot = os.path.join(fnDir, dbName)
        pattern = r'/Runs.*'
        fnDirBase = re.sub(pattern, "", fnDir)
        con = sqlite3.connect(dbRoot)
        print("Opened database successfully: ", dbRoot)
        # id = ID, Enabled = ENABLED, c67 = _xmipp_enhanced_psd, c01 = _defocusU, c02 = _defocusV, C03 = _defocusAngle
        # C65 = _xmipp_ctfVoltage
        query = "SELECT id, enabled, c67, c01, C02, C03, C65 from Objects"
        print('query: ', query)
        cursor = con.execute(query)

        for row in cursor:
            id = row[0]
            enabled = row[1]
            file = row[2]
            file = os.path.join(fnDirBase, file)
            dU = row[3]
            dV = row[4]
            dAngle = row[5]
            kV = row[6]
            dSinA = np.sin(2 * dAngle)
            dCosA = np.cos(2 * dAngle)

            if dataFlag == 1:
                if enabled == 1:
                    fileList.append((id, dU, dV, dSinA, dCosA, dAngle, kV, file))  # dmt
            else:
                fileList.append(file)

            if verbose == 1:
                print("ID = ", id)
                print("ENABLED", enabled)
                print("FILE = ", file)
                print("DEFOCUS_U = ", dU)
                print("DEFOCUS_V = ",  dV )
                print("DEFOCUS_ANGLE = ", dAngle )
                print("KV = ", kV, "\n")

        con.close()
        print("Closed database successfully")
        return fileList


    @staticmethod
    def downsampleCTF(fileList, stackDir, subset, dataFlag):
        if dataFlag == 1:
            cols = ['ID', 'DEFOCUS_U', 'DEFOCUS_V', 'Sin(2*angle)', 'Cos(2*angle)', 'Angle', 'kV', 'FILE']
            df_metadata = pd.DataFrame(fileList, columns=cols)
            df_metadata.insert(7, 'SUBSET', subset, True)

            for index in df_metadata.index.to_list():
                fnRoot = df_metadata.loc[index, 'FILE']
                fnBase = os.path.split(fnRoot)[1]  # name of the file
                destRoot = stackDir + fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset)
                shutil.copy(fnRoot, destRoot)
                df_metadata.loc[index, 'FILE'] = destRoot  #Change the name to the new one that is copied in this folder

            df_metadata.to_csv(os.path.join(stackDir, "metadata.csv"), index=False)
            print("Files copied to destiny and metadata generated")
            print(df_metadata)


        else:
            for fnRoot in fileList:
                fnBase = os.path.split(fnRoot)[1]
                destRoot = stackDir + fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset)
                df_metadata = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
                for index in df_metadata.index.to_list():
                    storedFile = df_metadata.loc[index, 'FILE']
                    storedFileBase = os.path.split(storedFile)[1]  # name of the file to store
                    if storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_1.xmp") \
                            or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_2.xmp") \
                            or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_3.xmp"):
                        shutil.copy(fnRoot, destRoot)
            print("Files copied to destiny")



    @staticmethod
    def prune(stackDir):
        metadataPath = open(os.path.join(stackDir, "metadata.txt"), "w+")
        lines = metadataPath.read().splitlines()
        lines.pop(0)
        nameFiles = []
        for line in lines:
            #fileName = line[40:-1] #está bien porque hay un espacio antes [88:-1]
            fileName = line[88:-1]
            nameFiles.append(fileName)

        for file in stackDir:
            for nameFile in nameFiles:
                nameFile.replace()
                if nameFile == file:
                    break
                file.replace(".xmp", "_ERRASE.xmp")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prepareTrainingDataset.py <dirIn> <dirOut> <subsetNumber> <importDataFlag(0/1)>")
        exit(0)

    fnDir = sys.argv[1]
    stackDir = sys.argv[2]
    subset = int(sys.argv[3])
    dataFlag = int(sys.argv[4])
    stackDir = stackDir + "/"
    deepDefocus = DeepDefocus()
    allPSDs = deepDefocus.importCTF(fnDir, dataFlag)
    deepDefocus.downsampleCTF(allPSDs, stackDir, subset, dataFlag)

    exit(0)
