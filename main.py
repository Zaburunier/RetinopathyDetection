import numpy as np

import cnn_mlp.train
import mtunet.train
import dffr_unet.train
from constants import RANDOM_SEED
import tensorflow as tf

import multiprocessing


def main():
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cnn_mlp.train.train_cnn_mlp()
    #mtunet.train.train_mtunet()
    #dffr_unet.train.train_dffr_unet()


def multiprocess_main():
    for i in range(100):
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    multiprocess_main()