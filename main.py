import numpy as np
import os
os.environ["CUDA_DIR"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"

from basemodel2.cnn import BuildKerasCNN
from basemodel2 import cnn_mlp
from data import DataGenerator
from constants import RANDOM_SEED, BATCH_SIZE, IMAGE_SIZE

from tensorflow.keras.losses import CategoricalHinge, CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf
from tensorboard import program
from losses.RecallLoss import MulticlassRecall
from losses.PrecisionLoss import MulticlassPrecision
from losses.F1Loss import MulticlassF1

import ctypes
ctypes.CDLL(r"C:\Windows\System32\vcomp140.dll")

import multiprocessing


def main():

    #print(tf.config.list_physical_devices('GPU'))
    tf.config.run_functions_eagerly(False)
    #tf.data.experimental.enable_debug_mode()
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    #for gpu in tf.config.list_physical_devices('GPU'):
    	#tf.config.experimental.set_memory_growth(gpu, True)

    model = BuildKerasCNN()

    trainDataset = DataGenerator.CreateDataset("D:/Study/Magister/CourseworkProject/main/data/img_trunc/train",
                                               "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainImgLabels.csv")
    testDataset = DataGenerator.CreateDataset("D:/Study/Magister/CourseworkProject/main/data/img_trunc/test",
                                              "D:\\Study\\Magister\\CourseworkProject\\main\\data\\testImgLabels.csv")
    # Веса посчитал вручную исходя из соотношения числа образцов в обучающей выборке
    #weights = {0: 0.6806, 1: 7.1915, 2: 3.3041, 3: 20.582, 4: 24.7759}
    weights = np.zeros(shape = (5, ), dtype = np.float32)
    weights[0] = len([filename for filename in os.listdir(".\\data\\img\\train\\0. No DR")])
    weights[1] = len([filename for filename in os.listdir(".\\data\\img\\train\\1. Mild DR")])
    weights[2] = len([filename for filename in os.listdir(".\\data\\img\\train\\2. Moderate DR")])
    weights[3] = len([filename for filename in os.listdir(".\\data\\img\\train\\3. Severe DR")])
    weights[4] = len([filename for filename in os.listdir(".\\data\\img\\train\\4. Profilerative DR")])
    totalAmount = np.sum(weights)
    #weights = weights / np.linalg.norm(weights, 2)
    weights = 0.5 * totalAmount / weights
    weights = {key : value for key, value in zip([0, 1, 2, 3, 4], weights)}
    #for instance in trainDataset.as_numpy_iterator():
    #    dummyData = tf.expand_dims(instance[0], axis = 0)#tf.data.Dataset.from_tensor_slices(tf.expand_dims(instance[0], axis = 0))
    #    dummyBool = True

    # В классе модели CNNMLP сейчас есть метрика CategoricalAccuracy
    model.compile(loss=CategoricalCrossentropy(from_logits = False),#MulticlassF1(nClasses = 5, class_weights = list(weights.values())), #CategoricalHinge(),
    			  jit_compile=False,
                  steps_per_execution = 200, metrics=[])
    #model.run_eagerly = True
    
    model.fit(trainDataset, verbose = 2, epochs = 40, workers = 2, use_multiprocessing = False, validation_data = testDataset, class_weight = weights)#, steps_per_epoch = 20, validation_steps = 20)

    result = model.predict(trainDataset.shuffle(200).batch(BATCH_SIZE).take(1))
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