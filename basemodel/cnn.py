# Обучаем составную модель из конволюшки в Keras и машины опорных векторов в SKLearn
import tensorflow as tf
import tensorflow.keras as tfkeras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, LeakyReLU, MaxPool2D, Dropout, Dense
#from keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.initializers import Zeros, RandomNormal, Constant, GlorotUniform, GlorotNormal, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization, Rescaling, RandomCrop, RandomFlip

import basemodel.cnn_mlp
from constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

def BuildKerasCNN():
    '''
    Метод создания конволюшки для выделения признаков (на время отладки цикла обучения здесь полная сеть)
    :return:
    '''
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size = BATCH_SIZE, dtype = tf.float32, name="input")
    crop = RandomCrop(224, 224) (inputs)
    flip = RandomFlip() (crop)
    rescaling = Rescaling(scale = 1.0 / 127.5, offset = -1.0) (flip)
    # Названия слоёв здесь соответствуют таблице с архитектурой сети из статьи

    # Первая свёртка. Входные д-е 256х256х3, окно размером 5х5, выходные д-е 256х256х64
    conv_1 = Conv2D(filters=64,
                    kernel_size=(5, 5),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_1",
                    kernel_initializer = HeNormal(),#RandomNormal(mean = 0.0, stddev = 1e-01, seed = RANDOM_SEED),
                    #bias_initializer = Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (rescaling)
    relu_1 = LeakyReLU(alpha = 1e-03, name = "relu_1") (conv_1)
    bn_1 = BatchNormalization(name="bn_1", epsilon = 1.0e-05) (relu_1)
    # Сокращаем размерность до 128х128х64
    mp_1 = MaxPool2D(pool_size = (2, 2), name = "mp_1") (bn_1)

    # Вторая свёртка. Входные д-е 128х128х64, окно размером 5х5, выходные д-е 128х128х64
    conv_2 = Conv2D(filters=64,
                    kernel_size=(5, 5),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_2",
                    kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                    #bias_initializer=Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (mp_1)
    relu_2 = LeakyReLU(alpha=1e-02, name = "relu_2")(conv_2)
    bn_2 = BatchNormalization(name="bn_2", epsilon=1.0e-05) (relu_2)
    # Сокращаем размерность до 64x64х64
    mp_2 = MaxPool2D(pool_size=(2, 2), name = "mp_2") (bn_2)

    # Третья свёртка. Входные д-е 64х64х64, окно размером 5х5, выходные д-е 64х64х128
    conv_3 = Conv2D(filters=128,
                    kernel_size=(5, 5),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_3",
                    kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                    #bias_initializer=Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (mp_2)
    relu_3 = LeakyReLU(alpha=1e-02, name="relu_3")(conv_3)
    bn_3 = BatchNormalization(name="bn_3", epsilon=1.0e-05)(relu_3)
    # Сокращаем размерность до 32х32х128
    mp_3 = MaxPool2D(pool_size=(2, 2), name="mp_3") (bn_3)

    # Четвёртая свёртка. Входные д-е 32х32х128, окно размером 3х3, выходные д-е 32х32х256
    conv_4 = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_4",
                    kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                    #bias_initializer=Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (mp_3)
    relu_4 = LeakyReLU(alpha=1e-02, name="relu_4")(conv_4)
    bn_4 = BatchNormalization(name="bn_4", epsilon=1.0e-05)(relu_4)
    # Сокращаем размерность до 16х16х256
    mp_4 = MaxPool2D(pool_size=(2, 2), name="mp_4") (bn_4)

    # Пятая свёртка. Входные д-е 16х16х256, окно размером 3х3, выходные д-е 16х16х512
    conv_5 = Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_5",
                    kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                    bias_initializer=Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (mp_4)
    relu_5 = LeakyReLU(alpha=1e-02, name="relu_5")(conv_5)
    bn_5 = BatchNormalization(name="bn_5", epsilon=1.0e-05)(relu_5)
    # Сокращаем размерность до 8х8х512
    mp_5 = MaxPool2D(pool_size=(2, 2), name="mp_5") (bn_5)

    # Шестая свёртка. Входные д-е 8х8х512, окно размером 3х3, выходные д-е 8х8х1024
    conv_6 = Conv2D(filters=1024,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    activation=None,
                    name="conv_6",
                    kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                    bias_initializer=Zeros(),
                    kernel_regularizer=L2(l2 = 1e-02),
                    ) (mp_5)
    relu_6 = LeakyReLU(alpha=1e-02, name="relu_6")(conv_6)
    bn_6 = BatchNormalization(name="bn_6", epsilon=1.0e-05)(relu_6)
    # Сокращаем размерность до 4х4х1024
    mp_6_1 = MaxPool2D(pool_size=(2, 2), name="mp_6_1", padding = "same") (bn_6)
    # Сокращаем размерность до 2x2х1024
    mp_6_2 = MaxPool2D(pool_size=(2, 2), name="mp_6_2")(mp_6_1)

    flatten = Flatten(name = "flatten") (mp_6_2)

    fc_1 = Dense(1024, name="fc_1",
                 activation = "elu",
                 kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                 bias_initializer=Zeros(),
                 kernel_regularizer=L2(l2 = 1e-02),
                 ) (flatten)
    dropout = Dropout(0.5, name = "dropout") (fc_1)
    fc_2 = Dense(32, name="fc_2",
                 activation = "softplus",
                 kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                 bias_initializer=Zeros(),
                 kernel_regularizer=L2(l2 = 1e-02),
                 ) (dropout)

    mlp_predictor = Dense(5,
                          name="fc_predict",
                          activation="softmax",
                          kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED)#,
                          #bias_initializer=Constant(1),
                          kernel_regularizer=L2(l2 = 1e-02),
                          ) (fc_2)

    result = basemodel.cnn_mlp.CNNMLP(inputs = inputs, outputs = mlp_predictor, name = "cnn_mlp")
    return result

