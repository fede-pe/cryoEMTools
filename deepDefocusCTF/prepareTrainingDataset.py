import os
import sys
import numpy as np
import re
import shutil
import sqlite3
import pandas as pd
from utils import rotation, sum_angles
import matplotlib.pyplot as plt
import xmippLib as xmipp

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
            destRoot = os.path.join(stackDir, fnBase)
            #shutil.copy(fnRoot, destRoot)
            img = xmipp.Image(fnRoot)
            img.convertPSD()
            img.write(destRoot)

            df_metadata.loc[index, 'FILE'] = destRoot  # Change the name to the new one that is copied in this folder

        if os.path.exists(os.path.join(stackDir, "metadata.csv")):
            df_prev = pd.read_csv(os.path.join(stackDir, "metadata.csv"))
            df_metadata = pd.concat([df_prev, df_metadata], ignore_index=True)

        df_metadata.to_csv(os.path.join(stackDir, "metadata.csv"), index=False)
        print("Files copied to destiny and metadata generated")
        print(df_metadata)

    print("Files copied to destiny")

def studyDataFrame(df, num_bins=10):
    defocus_column = df['DEFOCUS_U']
    # Create bins using cut
    bins = pd.cut(defocus_column, bins=num_bins)
    # Count the number of samples in each bin
    bin_counts = bins.value_counts()
    # Find the bin with the maximum count
    max_count_bin = bin_counts.idxmax()
    max_count = bin_counts[max_count_bin]
    # Calculate the number of cases each of the other bins needs to add
    additional_cases_needed = max_count - bin_counts
    result = {}
    i = 1
    for interval, additional_cases_needed in additional_cases_needed.items():
        result['interval'+str(i)] = {'interval_left': interval.left, 'interval_right': interval.right,
                                     'extra_cases': additional_cases_needed}
        i += 1

    print(result)
    return result

def augmentate_entries(entries_in_interval, number_extra_cases):
    if len(entries_in_interval) >= number_extra_cases:
        print('number of entries in interval bigger than extra cases needed')
        random_entries = entries_in_interval.sample(number_extra_cases)
        for index, row in random_entries.iterrows():
            file_path = random_entries.loc[index]['FILE']
            directory, file_name = os.path.split(file_path)
            angle_rotation = 90
            rotatedImage = rotation(file_path, angle_rotation)
            file_name_rot = str(angle_rotation) + '_' + file_name
            new_file_path = os.path.join(directory, file_name_rot)
            rotatedImage.write(new_file_path)
            random_entries.at[index, 'FILE'] = new_file_path
            real_angle = random_entries.loc[index]['Angle']
            new_angle = sum_angles(real_angle, angle_rotation)
            random_entries.at[index, 'Angle'] = new_angle
            random_entries.at[index, 'Sin(2*angle)'] = np.sin(2 * new_angle)
            random_entries.at[index, 'Cos(2*angle)'] = np.cos(2 * new_angle)
    else:
        print('number of extra cases bigger than number in interval')
        dataframes_to_concat = []
        if int(number_extra_cases/len(entries_in_interval)) < 2:
            print('extra_cases/entries < 2')
            number_extra_rotations = number_extra_cases - len(entries_in_interval)
            print('number of extra rotations ' + str(number_extra_rotations))
            random_entries = entries_in_interval.copy()
            for index, row in entries_in_interval.iterrows():
                file_path = entries_in_interval.loc[index]['FILE']
                directory, file_name = os.path.split(file_path)
                if number_extra_rotations > 0:
                    first = True
                    for angle_rotation in range(45, 91, 45):
                        rotatedImage = rotation(file_path, angle_rotation)
                        file_name_rot = str(angle_rotation) + '_' + file_name
                        new_file_path = os.path.join(directory, file_name_rot)
                        rotatedImage.write(new_file_path)
                        real_angle = entries_in_interval.loc[index]['Angle']
                        new_angle = sum_angles(real_angle, angle_rotation)

                        if first:
                            random_entries.at[index, 'FILE'] = new_file_path
                            random_entries.at[index, 'Angle'] = new_angle
                            random_entries.at[index, 'Sin(2*angle)'] = np.sin(2 * new_angle)
                            random_entries.at[index, 'Cos(2*angle)'] = np.cos(2 * new_angle)
                            first = False
                        else:
                            # Select the row you want to duplicate (e.g., the first row)
                            row_to_duplicate = entries_in_interval.loc[index].copy()
                            # Append the selected row to the DataFrame
                            row_to_duplicate['FILE'] = new_file_path
                            row_to_duplicate['Angle'] = new_angle
                            row_to_duplicate['Sin(2*angle)'] = np.sin(2 * new_angle)
                            row_to_duplicate['Cos(2*angle)'] = np.cos(2 * new_angle)
                            dataframes_to_concat.append(row_to_duplicate)

                    number_extra_rotations = number_extra_rotations - 1
                else:
                    angle_rotation = 90
                    rotatedImage = rotation(file_path, angle_rotation)
                    file_name_rot = str(angle_rotation) + '_' + file_name
                    new_file_path = os.path.join(directory, file_name_rot)
                    rotatedImage.write(new_file_path)
                    random_entries.at[index, 'FILE'] = new_file_path
                    real_angle = random_entries.loc[index]['Angle']
                    new_angle = sum_angles(real_angle, angle_rotation)
                    random_entries.at[index, 'Angle'] = new_angle
                    random_entries.at[index, 'Sin(2*angle)'] = np.sin(2 * new_angle)
                    random_entries.at[index, 'Cos(2*angle)'] = np.cos(2 * new_angle)

        else:
            print('extra_cases/entries > 2')
            extra_rotations = number_extra_cases / len(entries_in_interval)
            print('Number of extra rotations ' + str(extra_rotations))
            random_entries = entries_in_interval.copy()
            for index, row in entries_in_interval.iterrows():
                file_path = entries_in_interval.loc[index]['FILE']
                directory, file_name = os.path.split(file_path)
                first = True
                for angle_rotation in range(int(180 / extra_rotations), 181, int(180 / extra_rotations)):
                    rotatedImage = rotation(file_path, angle_rotation)
                    file_name_rot = str(angle_rotation) + '_' + file_name
                    new_file_path = os.path.join(directory, file_name_rot)
                    rotatedImage.write(new_file_path)
                    real_angle = entries_in_interval.loc[index]['Angle']
                    new_angle = sum_angles(real_angle, angle_rotation)

                    if first:
                        random_entries.loc[index, 'FILE'] = new_file_path
                        random_entries.loc[index, 'Angle'] = new_angle
                        random_entries.loc[index, 'Sin(2*angle)'] = np.sin(2 * new_angle)
                        random_entries.loc[index, 'Cos(2*angle)'] = np.cos(2 * new_angle)
                        first = False
                    else:
                        # Select the row you want to duplicate (e.g., the first row)
                        row_to_duplicate = entries_in_interval.loc[index].copy()
                        # Append the selected row to the DataFrame
                        row_to_duplicate['FILE'] = new_file_path
                        row_to_duplicate['Angle'] = new_angle
                        row_to_duplicate['Sin(2*angle)'] = np.sin(2 * new_angle)
                        row_to_duplicate['Cos(2*angle)'] = np.cos(2 * new_angle)
                        dataframes_to_concat.append(row_to_duplicate)

        new_entries = pd.DataFrame(dataframes_to_concat)
        random_entries = pd.concat([random_entries, new_entries], ignore_index=True)

    return random_entries

def generateData(dirOut):
    if os.path.exists(os.path.join(dirOut, "metadata.csv")):
        df = pd.read_csv(os.path.join(dirOut, "metadata.csv"))
        dict_intervals = studyDataFrame(df, num_bins=10)
        df_defocus_1 = df[['DEFOCUS_U', 'DEFOCUS_V']]
        df_defocus_1.plot.hist(alpha=0.5, bins=10)
        plt.show()
        result_df = df

        for dict in dict_intervals.values():
            # Define the defocus interval
            target_interval = (dict['interval_left'], dict['interval_right'])
            number_extra_cases = dict['extra_cases']
            defocus_column = df['DEFOCUS_U']
            print(target_interval, number_extra_cases)

            if number_extra_cases != 0:
                # Filter entries within the specified defocus interval
                entries_in_interval = df[(defocus_column >= target_interval[0]) & (defocus_column <= target_interval[1])]
                df_new_data = augmentate_entries(entries_in_interval, number_extra_cases)
                print(df_new_data)
                # Concatenate DataFrames with a different index
                result_df = pd.concat([result_df, df_new_data], ignore_index=True)
                print()
                print('Len new dataframe ' + str(len(result_df)))

        print(result_df)
        df_defocus = result_df['DEFOCUS_U']
        df_defocus.plot.hist(alpha=0.5, bins=10)
        plt.show()
        result_df.to_csv(os.path.join(dirOut, "metadata.csv"), index=False)
    else:
        print('There is no metadata file')

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python prepareTrainingDataset.py <dirIn> <dirOut> <subsetNumber> <importDataFlag(0/1)> <balanceDataset(0/1)>")
        exit(0)

    fnDir = sys.argv[1]
    dirOut = sys.argv[2]
    subset = int(sys.argv[3])
    dataFlag = int(sys.argv[4])
    balanceDataset = int(sys.argv[5])

    if balanceDataset != 1:
        allPSDs = importCTF(fnDir, dataFlag, useGroundTruth=False)
        createMetadataCTF(allPSDs, dirOut, subset, dataFlag, useGroundTruth=False)

    else:
        print('Balancing dataset with data augmentation')
        generateData(dirOut)

    exit(0)