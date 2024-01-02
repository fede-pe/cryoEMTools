import os
import sys
import numpy as np
import re
import shutil
import sqlite3
import pandas as pd

DB_NAME = 'ctfs.sqlite'
GROUND_TRUTH_PATH = '/home/dmarchan/data_hilbert_tres/TestNewPhantomData/simulationParameters.txt'


def loadGroundTruthValues(fnGT):
    """ Use to load a .txt with the ground truth values of the simulated CTFs """
    print('Loading ground truth defocus values')
    df = pd.read_csv(fnGT, sep=" ", header=None, index_col=False)
    df.columns = ["COUNTER", "FILE", "DEFOCUS_U", "DEFOCUS_V", "Angle"]
    return df

def getDefocusAnglesGT(df, entryFn):
    ''' Based on a Filename get the ground truth values for that image '''
    entryFn = os.path.basename(entryFn)
    end = entryFn.find("_xmipp")
    entryFn = entryFn[:end]
    found_entry = df[df["FILE"].str.contains(entryFn)]
    # print(found_entry)
    dU = found_entry['DEFOCUS_U'].values
    dV = found_entry['DEFOCUS_V'].values
    angle = found_entry['Angle'].values
    return dU[0], dV[0], angle[0]

def importCTF(fnDir, dataFlag, useGroundTruth):
    ''' Import the CTF information and PSD images from xmipp ctf estimation results or ground truth values '''
    fileList = []
    verbose = 1
    #CONNECT TO THE DATABASE
    dbName = DB_NAME
    dbRoot = os.path.join(fnDir, dbName)
    pattern = r'/Runs.*'
    fnDirBase = re.sub(pattern, "", fnDir)
    con = sqlite3.connect(dbRoot)
    print("Opened database successfully: ", dbRoot)
    # id = ID, Enabled = ENABLED, c05 = _psdFile, c01 = _defocusU, c02 = _defocusV, C03 = _defocusAngle
    # C65 = _xmipp_ctfVoltage
    query = "SELECT id, enabled, c05, c01, C02, C03, C58 from Objects"
    print('query: ', query)
    cursor = con.execute(query)

    if useGroundTruth:
        df_gt = loadGroundTruthValues(fnGT=GROUND_TRUTH_PATH)

    for row in cursor:
        id = row[0]
        enabled = row[1]
        file = row[2]
        file = os.path.join(fnDirBase, file)
        if useGroundTruth:
            dU, dV, dAngle = getDefocusAnglesGT(df_gt, file)
            dU_estimated = row[3]
            dV_estimated = row[4]
            dAngle_estimated = row[5]
        else:
            dU = row[3]
            dV = row[4]
            dAngle = row[5]

        dSinA = np.sin(2 * dAngle)
        dCosA = np.cos(2 * dAngle)
        kV = row[6]
        if dataFlag == 1:
            if useGroundTruth:
                fileList.append((id, dU, dV, dSinA, dCosA, dAngle, kV, file,
                                 dU_estimated, dV_estimated, dAngle_estimated))
            else:
                fileList.append((id, dU, dV, dSinA, dCosA, dAngle, kV, file))
        else:
            fileList.append(file)
        if verbose == 1:
            print("ID = ", id)
            print("ENABLED", enabled)
            print("FILE = ", file)
            print("DEFOCUS_U = ", dU)
            print("DEFOCUS_V = ",  dV)
            print("DEFOCUS_ANGLE = ", dAngle)
            print("KV = ", kV, "\n")

    con.close()
    print("Closed database successfully")
    return fileList

def createMetadataCTF(fileList, stackDir, subset, dataFlag, useGroundTruth):
    ''' Create the metadata file for training '''
    if dataFlag == 1:
        if useGroundTruth:
            cols = ['ID', 'DEFOCUS_U', 'DEFOCUS_V', 'Sin(2*angle)', 'Cos(2*angle)', 'Angle', 'kV', 'FILE',
                    'DEFOCUS_U_Est', 'DEFOCUS_V_Est', 'Angle_Est']
        else:
            cols = ['ID', 'DEFOCUS_U', 'DEFOCUS_V', 'Sin(2*angle)', 'Cos(2*angle)', 'Angle', 'kV', 'FILE']

        df_metadata = pd.DataFrame(fileList, columns=cols)
        df_metadata.insert(7, 'SUBSET', subset, True)
        for index in df_metadata.index.to_list():
            fnRoot = df_metadata.loc[index, 'FILE']
            fnBase = os.path.split(fnRoot)[1]  # name of the file
            destRoot = os.path.join(stackDir, fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset))
            shutil.copy(fnRoot, destRoot)
            df_metadata.loc[index, 'FILE'] = destRoot  # Change the name to the new one that is copied in this folder
        if os.path.exists(os.path.join(stackDir, "metadata.csv")):
            df_prev = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
            df_metadata = pd.concat([df_prev, df_metadata], ignore_index=True)

        df_metadata.to_csv(os.path.join(stackDir, "metadata.csv"), index=False)
        print("Files copied to destiny and metadata generated")
        print(df_metadata)
    else:
        for fnRoot in fileList:
            fnBase = os.path.split(fnRoot)[1]
            destRoot = os.path.join(stackDir, fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_%d.xmp" % subset))
            df_metadata = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
            for index in df_metadata.index.to_list():
                storedFile = df_metadata.loc[index, 'FILE']
                storedFileBase = os.path.split(storedFile)[1]  # name of the file to store
                if storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_1.xmp") \
                        or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_2.xmp") \
                        or storedFileBase == fnBase.replace("_xmipp_ctf_enhanced_psd.xmp", "_psdAt_3.xmp"):
                    shutil.copy(fnRoot, destRoot)
        print("Files copied to destiny")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prepareTrainingDataset.py <dirIn> <dirOut> <subsetNumber> <importDataFlag(0/1)>")
        exit(0)

    fnDir = sys.argv[1]
    dirOut = sys.argv[2]
    subset = int(sys.argv[3])
    dataFlag = int(sys.argv[4])
    allPSDs = importCTF(fnDir, dataFlag, useGroundTruth=False)
    createMetadataCTF(allPSDs, dirOut, subset, dataFlag, useGroundTruth=False)

    exit(0)