#import mtunet_builder
import constants
from .mtunet_builder  import BuildMTUnet
from data import SegmentationDataGenerator

import tensorflow as tf
import tensorflow_addons as tfa


def train_mtunet():
    mtunet = BuildMTUnet()
    #mtunet.use_grade_loss = True
    mtunet.SetupSegmentationMode()

    trainData, testData = SegmentationDataGenerator.CreateSegmentationDatasets("D:/Study/Magister/CourseworkProject/main/data/idrid/")

    mtunet.compile(jit_compile = False, metrics=[])

    '''
    nEpochs = 200
    mtunet.fit(currentLayer=trainData[:, :, :, :3], y=trainData[:, :, :, 3:], batch_size=constants.BATCH_SIZE,
               callbacks=[mtunet.csv_logger, mtunet.lr_logger],
               verbose=2, initial_epoch=mtunet.epoch_counter.read_value().numpy(),
               epochs=mtunet.epoch_counter.read_value().numpy() + nEpochs, workers=2,
               validation_data=(testData[:, :, :, :3], testData[:, :, :, 3:]))
    '''

    trainDataset = tf.data.Dataset.from_tensor_slices(trainData, "train")
    testDataset = tf.data.Dataset.from_tensor_slices(testData, "test")

    trainDataset = trainDataset.shuffle(trainDataset.cardinality(), reshuffle_each_iteration=True)
    testDataset = testDataset.shuffle(testDataset.cardinality(), reshuffle_each_iteration=True)

    trainDataset = trainDataset.repeat(1)
    testDataset = testDataset.repeat(1)

    trainDataset = trainDataset.map(AugmentData, tf.data.AUTOTUNE)
    testDataset = testDataset.map(AugmentData, tf.data.AUTOTUNE)

    trainDataset = trainDataset.batch(constants.BATCH_SIZE)
    testDataset = testDataset.batch(constants.BATCH_SIZE)

    trainDataset = trainDataset.shuffle(trainDataset.cardinality(), reshuffle_each_iteration=True)
    testDataset = testDataset.shuffle(testDataset.cardinality(), reshuffle_each_iteration=True)

    trainDataset = trainDataset.prefetch(4)
    testDataset = testDataset.prefetch(4)

    nEpochs = 200
    mtunet.fit(x=trainDataset, callbacks=[mtunet.csv_logger, mtunet.lr_logger],
               verbose=2, initial_epoch=mtunet.epoch_counter.read_value().numpy(),
               epochs=mtunet.epoch_counter.read_value().numpy() + nEpochs, workers=4,
               validation_data=testDataset)

    mtunet.epoch_counter.assign_add(nEpochs)
    mtunet.checkpoint_manager.save()


def main():
    train_mtunet()


if __name__ == "__main__":
    main()


@tf.function(jit_compile=True)
def AugmentData(data):
    '''
    Аугментация данных. Проводится здесь, а не в слоях нейросети, т. к. маски сегментации нужно аугментировать синхронно с исходным изображением
    :return:
    '''
    x = data[:, :, :3]
    y = data[:, :, 3:]
    segmentation = y

    tempXY = tf.concat([x, segmentation], axis=-1)

    # Аугментация 1: Зеркальное отражение по каждой оси
    tempXY = tf.image.random_flip_left_right(tempXY)
    tempXY = tf.image.random_flip_up_down(tempXY)

    # Аугментация 2: масштабирование (зум)
    #tempXY = self.imgZoomLayer(tempXY)

    # Аугментация 3: сдвиг
    shiftDirections = tf.random.uniform(shape=(2,), minval=-.1, maxval=.1)
    tempXY = tfa.image.translate(tempXY, shiftDirections)

    x, segmentation = tempXY[:, :, :3], tempXY[:, :, 3:]

    # Аугментация 4: Подмена цветовых каналов
    # channelOrder = tf.random.shuffle(tf.constant([0, 1, 2]))
    # currentLayer = tf.stack([currentLayer[:, :, channelOrder[0]], currentLayer[:, :, channelOrder[1]], currentLayer[:, :, channelOrder[2]]], axis = -1)

    y = segmentation

    result = tf.concat([x, y], axis = -1)

    return result