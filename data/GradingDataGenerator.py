import numpy as np
import pandas as pd

import os
import tensorflow as tf
import tensorflow.python.keras.utils as kerasutils
from tensorflow.python.keras.utils.data_utils import Sequence
from keras_preprocessing.image import array_to_img, img_to_array
from math import ceil
#from tensorflow import convert_to_tensor
import constants
from constants import IMAGE_SIZE, BATCH_SIZE
from PIL import Image

class DataGenerator(Sequence):
    '''
    Класс для подачи данных в сети.
    Через этот класс датасет загружается по частям, чтобы не держать все изображения в памяти сразу
    '''


    def __init__(self, image_filenames: [str] = None, label_csv_filename: str = None, batch_size: int = 32):
        '''
        Конструктор хранилища данных
        :param image_filenames: Список путей до изображения
        :param label_csv_filename: Путь до файла с разметкой классов
        :param batch_size: Размер одного бэтча
        '''
        self.batchSize = batch_size
        self.length = len(image_filenames)
        self.nBatches = ceil(self.length // self.batchSize)

        # Изображения хранятся в папке, а их классы - в отдельном CSV-файле
        self.images = np.zeros([self.length, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        self.labels = np.zeros([self.length])
        labelFile = pd.read_csv(label_csv_filename)

        for i in range(self.length):
            filename = image_filenames[i]
            self.images[i, :, :, :] = np.asarray(Image.open(filename))
            self.labels[i] = labelFile.loc[labelFile['image'] == filename, "segmentation"]



    def __getitem__(self, index):
        '''
        Получение бэтча с изображениями
        :param index: Индекс бэтча
        :return: Набор изображений в формате tf.Tensor (по осям: )
        '''

        pass


    def __len__(self):
        pass


    def on_epoch_end(self):
        # Перемешиваем данные
        newIndices = np.random.permutation(self.length)
        self.images = self.images[newIndices, :, :, :]
        self.labels = self.labels[newIndices]
        pass


def CreateGradingDataset(image_directory: [str] = None, label_csv_filename: str = None):
    labelFile = pd.read_csv(label_csv_filename)
    labels = np.reshape(labelFile.to_numpy()[:, 1], [-1, 1])

    dataset = tf.keras.utils.image_dataset_from_directory(image_directory, labels="inferred", label_mode = "categorical", image_size=IMAGE_SIZE, batch_size = None, shuffle = True, seed = constants.RANDOM_SEED)
    # С inferred labels объединение с лэйблами не нужно
    #dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(labels.astype(np.float32)))))
    #print(*[i for i in dataset.take(1).as_numpy_iterator()])
    return dataset