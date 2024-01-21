import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, LeakyReLU, MaxPool2D, Dropout, Dense, Add, Lambda
from tensorflow.keras.initializers import Zeros, RandomNormal, Constant, GlorotUniform, GlorotNormal, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D, Activation, Multiply, Concatenate

import cnn_mlp.cnn_mlp
from constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED
import layers.DepthwiseMaxPooling
import layers.DepthwiseAveragePooling

def BuildCNNMLPModel(useBinaryOutput = False):
    '''
    Метод создания конволюшки для выделения признаков (на время отладки цикла обучения здесь полная сеть)
    :return:
    '''
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size = BATCH_SIZE, dtype = tf.float32, name="input")

    # 256x256
    layer = Conv2D(
        filters = 64,
        kernel_size = 5,
        strides = 2,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),  # RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=1e-03),
    ) (inputs)
    # 128x128
    layer = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),  # RandomNormal(mean=0.0, stddev=1e-01, seed=RANDOM_SEED),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=1e-03),
    )(inputs)

    # 64x64
    layer = BuildResidualConvBlock(layer, 64, 64, (3, 3), 2, "res_64_1")
    layer = BuildResidualConvBlock(layer, 64, 64, (3, 3), 1, "res_64_2")
    layer = BuildResidualConvBlock(layer, 64, 64, (3, 3), 1, "res_64_3")

    # 32x32
    layer = BuildResidualConvBlock(layer, 64, 128, (3, 3), 2, "res_32_1")
    layer = BuildResidualConvBlock(layer, 128, 128, (3, 3), 1, "res_32_2")
    layer = BuildResidualConvBlock(layer, 128, 128, (3, 3), 1, "res_32_3")

    # 16x16
    layer = BuildResidualAttBlock(layer, 128, 256, (3, 3), 2, "att_16_1")
    layer = BuildResidualAttBlock(layer, 256, 256, (3, 3), 1, "att_16_2")

    # 8x8
    layer = BuildResidualAttBlock(layer, 256, 512, (3, 3), 2, "att_8_1")
    layer = BuildResidualAttBlock(layer, 512, 512, (3, 3), 1, "att_8_2")


    layer = GlobalMaxPooling2D() (layer)
    if useBinaryOutput:
        layer = Dense(1,
                      name="fc_predict_binary",
                      activation="sigmoid",
                      kernel_initializer=HeNormal(),
                      bias_initializer=Zeros(),
                      kernel_regularizer=L2(l2=1e-03),
                      bias_regularizer=L2(l2=1e-03),
                      )(layer)
    else:
        layer = Dense(5,
                      name="fc_predict",
                      activation="softmax",
                      kernel_initializer = HeNormal(),
                      bias_initializer=Zeros(),
                      kernel_regularizer=L2(l2 = 1e-03),
                      bias_regularizer=L2(l2 = 1e-03),
                      ) (layer)

    result = cnn_mlp.cnn_mlp.CNNMLP(useBinaryLabels = useBinaryOutput, inputs = inputs, outputs = layer, name ="cnn_mlp")
    return result


def BuildResidualConvBlock(currentLayer, currentFilters, endFilters, kernelSize, poolingSize = 1, baseName ="res"):
    currentLayer = BuildConvBlock(currentLayer, currentFilters, 1, 1, baseName = baseName + "_conv1")
    currentLayer = BuildConvBlock(currentLayer, currentFilters, kernelSize, poolingSize, baseName = baseName + "_dconv", depthwise = True)
    currentLayer = BuildConvBlock(currentLayer, endFilters, 1, 1, baseName = baseName + "_conv2")
    return currentLayer


def BuildResidualAttBlock(currentLayer, currentFilters, endFilters, kernelSize, poolingSize = 1, baseName ="att"):
    attention = BuildAttentionBlock(currentLayer, currentFilters, baseName + "_attention")
    currentLayer = Add(name = baseName + "_add") ([attention, currentLayer])
    currentLayer = LeakyReLU(alpha=1e-02, name = baseName + "_relu") (currentLayer)
    currentLayer = BatchNormalization(epsilon=1.001e-05, name = baseName + "_bn") (currentLayer)
    currentLayer = BuildConvBlock(currentLayer, endFilters, kernelSize, poolingSize, baseName = baseName + "_conv")
    return currentLayer


def BuildAttentionBlock(currentLayer, nFilters, baseName ="att"):
    attChannel = BuildChannelAttentionBlock(currentLayer, nFilters, baseName=baseName + "_ch")
    layer = Multiply(name = baseName + "_ch_mul") ([currentLayer, attChannel])

    attSpatial = BuildSpatialAttentionBlock(currentLayer, nFilters, baseName=baseName + "_sp")
    layer = Multiply(name = baseName + "_sp_mul") ([layer, attSpatial])
    return layer


def BuildChannelAttentionBlock(currentLayer, nFilters, baseName ="att_ch"):
    maxPoolLayer = GlobalMaxPooling2D(keepdims = True, name = baseName + "_max_pool") (currentLayer)
    avgPoolLayer = GlobalAveragePooling2D(keepdims=True, name = baseName + "_avg_pool") (currentLayer)

    denseNetwork = BuildChannelAttentionDenseBlock(nFilters, baseName + "_dense")

    densedMaxPoolLayer = denseNetwork(maxPoolLayer)
    densedAvgPoolLayer = denseNetwork(avgPoolLayer)

    densedLayer = Add(name = baseName + "_pool_add") ([densedMaxPoolLayer, densedAvgPoolLayer])
    sigmoidLayer = Activation("sigmoid", name = baseName + "_pool_act") (densedLayer)
    return sigmoidLayer


def BuildChannelAttentionDenseBlock(nFilters, baseName ="att_ch_dense"):
    denseModel = Sequential(name = baseName)
    denseModel.add(Input([1, 1, nFilters], batch_size = BATCH_SIZE, name = baseName + "_input"))
    denseModel.add(Dense(nFilters / 8, name = baseName + "_dense_inner"))
    denseModel.add(Dense(nFilters, name = baseName + "_dense_outer"))
    return denseModel


def BuildSpatialAttentionBlock(currentLayer, nFilters, baseName ="att_sp"):
    maxPoolLayer = layers.DepthwiseMaxPooling.DepthwiseMaxPooling(name=baseName + "_max_pool") (currentLayer)
    avgPoolLayer = layers.DepthwiseAveragePooling.DepthwiseAveragePooling(name=baseName + "_avg_pool") (currentLayer)
    poolLayer = Concatenate(name = baseName + "_pool_concat") ([maxPoolLayer, avgPoolLayer])
    convLayer = Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer=L2(l2=1e-03),
        name=baseName + "_pool_conv"
    )(poolLayer)
    sigmoidLayer = Activation("sigmoid", name = baseName + "_pool_act") (convLayer)
    return sigmoidLayer



def BuildConvBlock(currentLayer, nFilters, kernelSize, poolingSize, baseName, batchNorm = True, depthwise = False):
    if depthwise:
        layer = DepthwiseConv2D(
            depth_multiplier=poolingSize,
            kernel_size=3,
            strides=poolingSize,
            padding="same",
            activation=None,
            kernel_initializer=HeNormal(),
            bias_initializer=Zeros(),
            kernel_regularizer=L2(l2=1e-03),
            bias_regularizer=L2(l2=1e-03),
            name=baseName + "_dconv"
        ) (currentLayer)
    else:
        layer = Conv2D(
            filters=nFilters,
            kernel_size=kernelSize,
            strides=poolingSize,
            padding="same",
            activation=None,
            kernel_initializer=HeNormal(),
            bias_initializer=Zeros(),
            kernel_regularizer=L2(l2=1e-03),
            bias_regularizer=L2(l2=1e-03),
            name=baseName + "_conv"
        )(currentLayer)

    if batchNorm:
        layer = LeakyReLU(alpha=1e-02, name=baseName + "_relu")(layer)
        layer = BatchNormalization(epsilon=1.001e-05, name=baseName + "_bn")(layer)

    return layer




