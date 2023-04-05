import keras.preprocessing.image_dataset
import numpy as np
import os

from basemodel2.cnn import BuildKerasCNN
from basemodel2 import cnn_mlp
from data import DataGenerator
from constants import RANDOM_SEED, BATCH_SIZE, IMAGE_SIZE

import tensorflow as tf
from tensorboard import program
from losses.RecallLoss import MulticlassRecall
from losses.PrecisionLoss import MulticlassPrecision
from losses.F1Loss import MulticlassF1

import ctypes
ctypes.CDLL(r"C:\Windows\System32\vcomp140.dll")

import multiprocessing


def main():
    tf.config.run_functions_eagerly(False)
    #tf.data.experimental.enable_debug_mode()
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = BuildKerasCNN()

    trainDataset = DataGenerator.CreateDataset("D:/Study/Magister/CourseworkProject/main/data/img/train",
                                               "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainImgLabels.csv")
    testDataset = DataGenerator.CreateDataset("D:/Study/Magister/CourseworkProject/main/data/img/test",
                                              "D:\\Study\\Magister\\CourseworkProject\\main\\data\\testImgLabels.csv")
    # Веса посчитал вручную исходя из соотношения числа образцов в обучающей выборке
    weights = {0: 0.6806, 1: 7.1915, 2: 3.3041, 3: 20.582, 4: 24.7759}


    #for instance in trainDataset.take(1).as_numpy_iterator():
    #    dummyData = tf.expand_dims(instance[0], axis = 0)#tf.data.Dataset.from_tensor_slices(tf.expand_dims(instance[0], axis = 0))

    # В классе модели CNNMLP сейчас есть метрика CategoricalAccuracy
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True), jit_compile=True, #MulticlassF1(nClasses = 5, class_weights = list(weights.values())), jit_compile=False,
                  steps_per_execution = 10, metrics=[])
    #model.run_eagerly = True
    
    model.fit(trainDataset, verbose = 2, epochs = 10, workers = 2, use_multiprocessing = False, validation_data = testDataset, class_weight = weights)#, steps_per_epoch = 200, validation_steps = 20)

    result = model.predict(trainDataset.batch(BATCH_SIZE).take(1))
    print(result)


def multiprocess_main():
    for i in range(100):
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()


if __name__ == "__main__":
    #tracking_address = "D:\\logs"
    #tb = program.TensorBoard()
    #tb.configure(argv=[None, '--logdir', tracking_address])
    #url = tb.launch()
    #print(f"Tensorflow listening on {url}")

    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    multiprocess_main()
    #main()