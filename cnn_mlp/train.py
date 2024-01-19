import os
import numpy as np

#import cnn_mlp.cnn_imagenet
#import cnn_mlp.cnn_resnet
import cnn_mlp.cnn_attention
from data import GradingDataGenerator
from constants import RANDOM_SEED, BATCH_SIZE, IMAGE_SIZE, DATA_PATH, DATASET_PATH

from tensorflow.keras.losses import CategoricalHinge, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import RandomZoom, RandomTranslation
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import random_zoom
from tensorboard import program
from losses.RecallLoss import MulticlassRecall
from losses.PrecisionLoss import MulticlassPrecision
from losses.F1Loss import MulticlassF1


def train_cnn_mlp():
    useBinaryOutput = False
    model = cnn_mlp.cnn_attention.BuildCNNMLPModel(useBinaryOutput)
    #model.SetupImageNetWeights()

    trainDataset = GradingDataGenerator.CreateGradingDataset(DATASET_PATH + "train",
                                                      DATA_PATH + "trainImg2Labels.csv")
    testDataset = GradingDataGenerator.CreateGradingDataset(DATASET_PATH + "test",
                                                     DATA_PATH + "testImg2Labels.csv")

    trainSteps = -1
    testSteps = -1

    trainDataset = \
        trainDataset. \
            shuffle(128, reshuffle_each_iteration=True). \
            map(PrepareImage, tf.data.AUTOTUNE). \
            repeat(1). \
            batch(BATCH_SIZE, drop_remainder=True). \
            shuffle(32, reshuffle_each_iteration=True). \
            prefetch(tf.data.AUTOTUNE). \
            take(trainSteps)
    testDataset = \
            testDataset. \
            shuffle(128, reshuffle_each_iteration=True). \
            map(PrepareImage, tf.data.AUTOTUNE). \
            repeat(1). \
            batch(BATCH_SIZE, drop_remainder=True). \
            shuffle(32, reshuffle_each_iteration=True). \
            prefetch(tf.data.AUTOTUNE). \
            take(testSteps)

    # Считаем веса исходя из дизбаланса выборки (для увеличения градиентов там, где слишком мало образцов)
    weights = np.zeros(shape=(5,), dtype=np.float32)
    weights[0] = len([filename for filename in os.listdir(DATASET_PATH + "train/0. No DR")])
    weights[1] = len([filename for filename in os.listdir(DATASET_PATH + "train/1. Mild DR")])
    weights[2] = len([filename for filename in os.listdir(DATASET_PATH + "train/2. Moderate DR")])
    weights[3] = len([filename for filename in os.listdir(DATASET_PATH + "train/3. Severe DR")])
    weights[4] = len([filename for filename in os.listdir(DATASET_PATH + "train/4. Profilerative DR")])
    totalAmount = np.sum(weights)
    # weights = weights / np.linalg.norm(weights, 2)
    weights = 0.5 * totalAmount / weights
    weights = {key: value for key, value in zip([0, 1, 2, 3, 4], weights)}

    # В классе модели CNNMLP сейчас есть метрика CategoricalAccuracy
    if useBinaryOutput:
        model.compile(loss=BinaryCrossentropy(from_logits=False),
                      jit_compile=False,
                      steps_per_execution=1, metrics=[])
    else:
        model.compile(loss=CategoricalCrossentropy(from_logits=False),
                      jit_compile=False,
                      steps_per_execution=1, metrics=[])

    model.fit(trainDataset, verbose=2, epochs=40, validation_data=testDataset,
              class_weight=weights, steps_per_epoch = trainSteps if trainSteps > 0 else None, validation_steps = testSteps if testSteps > 0 else None)

    #result = model.predict(trainDataset.shuffle(200).batch(BATCH_SIZE).take(1))
    #print(result)


zoomLayer = RandomZoom(height_factor = (-0.1, 0.3), fill_mode = "constant", fill_value = 0.5)
translateLayer = RandomTranslation(0.2, 0.2, fill_mode = 'constant', fill_value = 0.5)


@tf.function(jit_compile = False)
def AugmentData(x, y):
    tempX = x

    # Аугментация 1: Зеркальное отражение по каждой оси
    tempX = tf.image.random_flip_left_right(tempX)
    tempX = tf.image.random_flip_up_down(tempX)

    # Аугментация 2: масштабирование (зум)
    tempX = zoomLayer(tempX, training = True)

    # Аугментация 3: сдвиг
    tempX = translateLayer(tempX, training = True)

    # Аугментация 4: модификация параметров изображения
    imageAugmentRatios = tf.random.uniform((4,), -1.0, 1.0)
    tempX = tf.image.adjust_brightness(tempX, .3 * imageAugmentRatios[0])
    tempX = tf.image.adjust_contrast(tempX, 1.0 + 0.3 * imageAugmentRatios[1])
    tempX = tf.image.adjust_saturation(tempX, 1.0 + 0.3 * imageAugmentRatios[2])
    tempX = tf.image.adjust_hue(tempX, imageAugmentRatios[3])

    max = tf.reduce_max(tempX)#, axis = (1, 2, 3), keepdims = True)
    min = tf.reduce_min(tempX)#, axis = (1, 2, 3), keepdims = True)

    tempX = (tempX - min) / (max - min)

    return tempX, y


@tf.function(jit_compile = False)
def PrepareImage(x, y):
    '''
    Подготовка изображения к подаче в нейронную сеть
    :param x: Исходное изображение сетчатки
    :return:
    '''
    x = x / 255.0
    return x, y

