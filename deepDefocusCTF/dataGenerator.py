import tensorflow as tf
import numpy as np
from utils import centerWindow

class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col, batch_size, input_size=(512, 512, 1), shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        imageMatrixNorm = centerWindow(path, objective_res=2, sampling_rate=1)

        return imageMatrixNorm

    def __get_output_U(self, defocusU):
        defocusVector = np.zeros(1, dtype=np.float64)
        defocusVector[0] = defocusU

        return defocusVector

    def __get_output_V(self, defocusV):
        defocusVector = np.zeros(1, dtype=np.float64)
        defocusVector[0] = defocusV

        return defocusVector

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        image_batch = batches[self.X_col['path']] # 'FILE' in our dataframe

        defocus_batch = batches[[self.y_col['defocus_U'], self.y_col['defocus_V']]]

        X_batch = np.asarray([self.__get_input(x) for x in image_batch])

        yU_batch = np.asarray([self.__get_output_U(y) for y in defocus_batch[self.y_col['defocus_U']]])
        yV_batch = np.asarray([self.__get_output_V(y) for y in defocus_batch[self.y_col['defocus_V']]])

        return X_batch, tuple([yU_batch, yV_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

class CustomDataGenPINN(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        imageMatrixNorm = centerWindow(path, objective_res=2, sampling_rate=1)
        #imageFiltered = gaussian_low_pass_filter(imageMatrixNorm)
        return imageMatrixNorm
        #return imageFiltered

    def __get_output_defocus(self, defocus_scaled):
        return np.expand_dims(np.array(defocus_scaled), axis=-1)
    def __get_output_angle(self, angle):
        return np.expand_dims(np.array(angle), axis=-1)

    # def __get_output_CTF_corr(self, CTF_corr_labels):
    #     return np.expand_dims(np.array(CTF_corr_labels), axis=-1)

    def __get_data(self, batches):
        image_batch = batches['FILE']
        defocus_U_batch_unscaled = batches['DEFOCUS_U_SCALED'].to_numpy()
        defocus_V_batch_unscaled = batches['DEFOCUS_V_SCALED'].to_numpy()
        defocus_angle_batch = batches['NORMALIZED_ANGLE'].to_numpy()

        X_batch = np.asarray([self.__get_input(x) for x in image_batch])
        # Ensure that all output arrays have the same number of dimensions
        yU_batch = np.asarray([self.__get_output_defocus(defocus_U) for defocus_U in defocus_U_batch_unscaled])
        yV_batch = np.asarray([self.__get_output_defocus(defocus_V) for defocus_V in defocus_V_batch_unscaled])
        yAngle_batch = np.asarray([self.__get_output_angle(angle) for angle in defocus_angle_batch])

        # Concatenate the four outputs
        y_batch = np.concatenate([yU_batch, yV_batch, yAngle_batch], axis=-1)

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))