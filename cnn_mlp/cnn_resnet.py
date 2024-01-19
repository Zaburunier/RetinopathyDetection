import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, LeakyReLU, MaxPool2D, Dropout, Dense, Add, Lambda
from tensorflow.keras.initializers import Zeros, RandomNormal, Constant, GlorotUniform, GlorotNormal, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D

import cnn_mlp
from constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

def BuildCNNMLPModel() -> cnn_mlp.CNNMLP:
    '''
    Метод создания конволюшки для выделения признаков (на время отладки цикла обучения здесь полная сеть)
    :return:
    '''
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size = BATCH_SIZE, dtype = tf.float32, name="input")

    # 256x256
    layer = Conv2D(
        filters = 64,
        kernel_size = 7,
        strides = 2,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),  # RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=1e-03),
    ) (inputs)
    # 128x128
    layer = MaxPool2D(pool_size=(2, 2)) (layer)

    # 64x64
    layer = BuildResidualBlock(layer, 64, 64, (3, 3), 1, "res_64_1")
    layer = BuildResidualBlock(layer, 64, 64, (3, 3), 1, "res_64_2")

    # 32x32
    layer = BuildResidualBlock(layer, 64, 128, (3, 3), 2, "res_32_1")
    layer = BuildResidualBlock(layer, 128, 128, (3, 3), 1, "res_32_2")
    layer = BuildResidualBlock(layer, 128, 128, (3, 3), 1, "res_32_3")

    # 16x16
    layer = BuildResidualBlock(layer, 128, 256, (3, 3), 2, "res_16_1")
    layer = BuildResidualBlock(layer, 256, 256, (3, 3), 1, "res_16_2")

    # 8x8
    layer = BuildResidualBlock(layer, 256, 512, (3, 3), 2, "res_8_1")
    layer = BuildResidualBlock(layer, 512, 512, (3, 3), 1, "res_8_2")

    layer = GlobalMaxPooling2D() (layer)
    layer = Dense(5,
                  name="fc_predict",
                  activation="softmax",
                  kernel_initializer = HeNormal(),
                  bias_initializer=Zeros(),
                  kernel_regularizer=L2(l2 = 1e-03),
                  bias_regularizer=L2(l2 = 1e-03),
                  ) (layer)

    result = cnn_mlp.CNNMLP(inputs = inputs, outputs = layer, name ="cnn_mlp")
    return result


def BuildResidualBlock(currentLayer, currentFilters, endFilters, kernelSize, poolingSize = 1, baseName = "res"):
    if poolingSize == 1:
        currentLayerForResLink = currentLayer
    else:
        currentLayerForResLink = Conv2D(
            filters=endFilters,
            kernel_size=kernelSize,
            strides=poolingSize,
            padding="same",
            activation=None,
            kernel_initializer=HeNormal(),
            bias_initializer=Zeros(),
            kernel_regularizer=L2(l2=1e-03),
            bias_regularizer=L2(l2=1e-03),
            name = baseName + "_skip_conv"
        )(currentLayer)
        currentLayerForResLink = BatchNormalization(epsilon=1.001e-05, name = baseName + "_skip_bn")(currentLayerForResLink)

    currentLayer = BuildResidualConvBlock(currentLayer, currentFilters, kernelSize, poolingSize, baseName + "_1")
    currentLayer = BuildResidualConvBlock(currentLayer, currentFilters, kernelSize, 1, baseName + "_2")
    currentLayer = BuildResidualConvBlock(currentLayer, endFilters, kernelSize, 1, baseName + "_3")
    currentLayer = Add(name = baseName + "_add") ([currentLayer, currentLayerForResLink])

    return currentLayer


def BuildResidualConvBlock(currentLayer, nFilters, kernelSize, poolingSize = 1, baseName = "res##"):
    currentLayer = Conv2D(
        filters=nFilters,
        kernel_size=kernelSize,
        strides=poolingSize,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=1e-03),
        name = baseName + "_conv"
    )(currentLayer)

    currentLayer = BatchNormalization(epsilon=1.001e-05, name = baseName + "_bn")(currentLayer)
    currentLayer = LeakyReLU(alpha=1e-02, name = baseName + "_relu")(currentLayer)

    return currentLayer


