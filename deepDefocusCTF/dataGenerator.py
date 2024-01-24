import tensorflow as tf
import numpy as np
import xmippLib as xmipp
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


class CustomDataGenAngle(tf.keras.utils.Sequence):

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
        img = xmipp.Image(path).getData()
        # Normalization
        imageMatrixNorm = (img - np.mean(img))/np.std(img)

        return imageMatrixNorm

    def __get_output_sinAngle(self, sinAngle):
        # TO DO use the outputLabel
        anglesVector = np.zeros(1, dtype=np.float64)
        anglesVector[0] = sinAngle

        return anglesVector

    def __get_output_cosAngle(self, cosAngle):
        # TO DO use the outputLabel
        anglesVector = np.zeros(1, dtype=np.float64)
        anglesVector[0] = cosAngle

        return anglesVector

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        image_batch = batches[self.X_col['path']]  # 'FILE' in our dataframe
        angle_batch = batches[[self.y_col['sinAngle'], self.y_col['cosAngle']]]
        X_batch = np.asarray([self.__get_input(x) for x in image_batch])
        sinAngle_batch = np.asarray([self.__get_output_sinAngle(y) for y in angle_batch[self.y_col['sinAngle']]])
        cosAngle_batch = np.asarray([self.__get_output_cosAngle(y) for y in angle_batch[self.y_col['cosAngle']]])

        return X_batch, tuple([sinAngle_batch, cosAngle_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size