import pandas as pd
import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D,\
    Lambda, Concatenate, Reshape, UpSampling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from utils import startSessionAndInitialize, make_training_plots
import matplotlib.pyplot as plt
import numpy as np
import xmippLib as xmipp
import datetime
import cv2
from skimage.metrics import structural_similarity as ssim


BATCH_SIZE = 8
EPOCHS = 100

class DenoisingAutoencoder:
    def __init__(self, modelDir='/', input_shape=(512, 512, 1), latent_dim=8):
        self.modelDir = modelDir
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = Input(shape=self.input_shape)

        # Encoder
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(inputs=input_img, outputs=decoded, name="denoising_autoencoder")
        autoencoder.summary()
        optimizer = Adam(learning_rate=0.001)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

        return autoencoder

    def train_autoencoder(self, train_generator, val_generator, epochs=10):
        path_logs_autoencoder = os.path.join(self.modelDir,
                                         "logs_autoencoder/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.modelDir = path_logs_autoencoder
        callbacks_list_def = [
            callbacks.CSVLogger(os.path.join(self.modelDir, 'autoencoder.csv'), separator=',', append=False),
            callbacks.TensorBoard(log_dir=self.modelDir, histogram_freq=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                        mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=20)
                            ]
        history = self.autoencoder.fit(train_generator, epochs=epochs,
                                       validation_data=val_generator, callbacks=callbacks_list_def)
        return history

    def denoise_images(self, input_data):
        return self.autoencoder.predict(input_data)

# def simple_data_generator(dataframe, batch_size=16):
#     while True:
#         indices = np.random.randint(0, len(dataframe), batch_size)
#         batch_files = dataframe.iloc[indices]['FILE'].values
#         batch_images = []
#         for file in batch_files:
#             img = xmipp.Image(file).getData()
#             # Normalization
#             imgNorm = (img - np.mean(img)) / np.std(img)
#             batch_images.append(img_norm)
#
#         batch_images = np.array(batch_images)
#         yield batch_images, batch_images  # Autoencoder input and target are the same

class PSDDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.indices = np.arange(len(dataframe))

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indices = self.indices[start:end]

        batch_images = []

        for idx in batch_indices:
            file = self.dataframe.iloc[idx]['FILE']
            img = xmipp.Image(file).getData()
            # Normalization
            imageNorm = (img - np.mean(img)) / np.std(img)
            batch_images.append(imageNorm)

        return np.array(batch_images), np.array(batch_images)  # Input and target are the same for denoising autoencoder

def psnr(original, denoised):
    """
    PSNR is a measure of the quality of an image.
    It's calculated as the ratio of the maximum possible power of a signal to the power of corrupting noise.
    """
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(original, denoised):
    """
    SSIM measures the structural similarity between two images.
    A higher SSIM indicates a better similarity.
    """
    ssim_value = ssim(original, denoised, data_range=np.max(original) - np.min(original))
    return ssim_value

# ------------------------ MAIN PROGRAM -----------------------------
if __name__ == "__main__":

    input_size = (512, 512, 1)
    # ----------- INITIALIZING SYSTEM ------------------
    # startSessionAndInitialize()

    # ----------- LOADING DATA ------------------
    print("Loading data...")
    metadataDir = '/home/dmarchan/TFM/TestNew'
    modelDir = '/home/dmarchan/TFM/TestNewAutoencoder'
    path_metadata = os.path.join(metadataDir, "metadata.csv")
    df_metadata = pd.read_csv(path_metadata)
    print(df_metadata.head())

    # ----------- SPLIT DATA: TRAIN, VALIDATE and TEST ------------
    df_train, df_test = train_test_split(df_metadata, test_size=0.20)

    df_train, df_validate = train_test_split(df_metadata, test_size=0.20)
    _, df_test = train_test_split(df_metadata, test_size=0.10)

    # ----------- TRAINING MODELS-------------------
    train_generator = PSDDataGenerator(df_train, batch_size=BATCH_SIZE)
    val_generator = PSDDataGenerator(df_validate, batch_size=BATCH_SIZE)
    test_generator = PSDDataGenerator(df_test, batch_size=1)

    # Instantiate the DenoisingAutoencoder class
    denoising_autoencoder = DenoisingAutoencoder(modelDir=modelDir, input_shape=(512, 512, 1), latent_dim=8)

    # Train the autoencoder on your noisy PSD images
    history = denoising_autoencoder.train_autoencoder(train_generator, epochs=EPOCHS, val_generator=val_generator)

    make_training_plots(history, denoising_autoencoder.modelDir, "autoencoder_")

    # Denoise PSD images using the trained autoencoder
    denoised_images = denoising_autoencoder.denoise_images(test_generator)

    # Select a few images for visualization
    original_images_list = []  # List to store original images
    denoised_images_list = []  # List to store denoised images
    psnr_values = []  # List to store PSNR values
    ssim_values = []  # List to store SSIM values

    num_images = 5
    selected_indices = np.random.choice(len(df_test), num_images, replace=False)

    # Assuming you have a loop where you load your images and denoise them
    for i, index in enumerate(selected_indices):
        path = df_test.iloc[index]['FILE']
        original_image = xmipp.Image(path).getData()
        denoised_image = denoised_images[index, :, :, 0]

        original_images_list.append(original_image)
        denoised_images_list.append(denoised_image)

        psnr_value = psnr(original_image, denoised_image)
        ssim_value = calculate_ssim(original_image, denoised_image)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    # Plot original and denoised PSD images
    plt.figure(figsize=(18, 10))
    for i in range(num_images):
        # Plot original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images_list[i], cmap='gray')
        plt.title(f'Original Image {selected_indices[i] + 1}')
        plt.axis('off')

        # Plot denoised image with PSNR value
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(denoised_images_list[i], cmap='gray')
        plt.title(f'Denoised Image {selected_indices[i] + 1}\nPSNR: {psnr_values[i]:.2f} dB\n'
                  f'SSIM: {ssim_values[i]:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(denoising_autoencoder.modelDir, "autoencoder_" + 'Training.png'))
    # plt.show()

