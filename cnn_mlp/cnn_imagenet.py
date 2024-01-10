# Обучаем составную модель из конволюшки в Keras и машины опорных векторов в SKLearn
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, LeakyReLU, MaxPool2D, Dropout, Dense
from tensorflow.keras.initializers import Zeros, RandomNormal, Constant, GlorotUniform, GlorotNormal, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization

import cnn_mlp.cnn_mlp
from constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

def BuildKerasCNN() -> cnn_mlp.cnn_mlp.CNNMLP:
    '''
    Метод создания конволюшки для выделения признаков (на время отладки цикла обучения здесь полная сеть)
    :return:
    '''
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size = BATCH_SIZE, dtype = tf.float32, name="input")

    conv_11 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_11",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(inputs)

    conv_12 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_12",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_11)

    mp_21 = MaxPool2D(pool_size=(2, 2), name="mp_21")(conv_12)

    conv_21 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_21",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(mp_21)

    conv_22 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_22",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_21)

    mp_31 = MaxPool2D(pool_size=(2, 2), name="mp_31")(conv_22)

    conv_31 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_31",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(mp_31)

    conv_32 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_32",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_31)

    conv_33 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_33",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_32)

    mp_41 = MaxPool2D(pool_size=(2, 2), name="mp_41")(conv_33)

    conv_41 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_41",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(mp_41)

    conv_42 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_42",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_41)

    conv_43 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_43",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_42)

    mp_51 = MaxPool2D(pool_size=(2, 2), name="mp_51")(conv_43)

    conv_51 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_51",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(mp_51)

    conv_52 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_52",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_51)

    conv_53 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_53",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=2e-03)
    )(conv_52)

    flatten = Flatten(name = "flatten") (conv_53)

    #fc_1 = Dense(1024, name="fc_1",
    #             activation = None,
    #             kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
    #             bias_initializer=Zeros(),
    #             kernel_regularizer=L2(l2 = 1e-03),
    #             bias_regularizer=L2(l2 = 1e-03),
    #             ) (flatten)
    #dropout = Dropout(0.5, name = "dropout") (fc_1)
    #relu_fc1 = LeakyReLU(alpha=1e-02, name="relu_fc1") (dropout)
    fc_2 = Dense(32, name="fc_2",
                 activation = None,
                 kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
                 bias_initializer=Zeros(),
                 kernel_regularizer=L2(l2 = 1e-03),
                 bias_regularizer=L2(l2 = 1e-03),
                 ) (flatten)
    relu_fc2 = LeakyReLU(alpha=1e-02, name="relu_fc2") (fc_2)

    mlp_predictor = Dense(5,
                          name="fc_predict",
                          activation="softmax",
                          kernel_initializer = HeNormal(),#RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED)#,
                          bias_initializer=Zeros(),
                          kernel_regularizer=L2(l2 = 1e-03),
                          bias_regularizer=L2(l2 = 1e-03),
                          ) (relu_fc2)

    result = cnn_mlp.cnn_mlp.CNNMLP(inputs = inputs, outputs = mlp_predictor, name ="cnn_mlp")
    return result

