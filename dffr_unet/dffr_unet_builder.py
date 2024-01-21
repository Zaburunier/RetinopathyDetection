from constants import IMAGE_SIZE, BATCH_SIZE

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalMaxPool2D, Conv2DTranspose, LeakyReLU, Add, Concatenate, Lambda, Activation
from tensorflow.keras.layers import Input, Rescaling, RandomCrop, RandomFlip, BatchNormalization, Multiply, Softmax, MultiHeadAttention, UpSampling2D
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Zeros, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2

#import dffr_unet
#import dffr_unet.dffr_unet
#import dffr_unet.RepeatFiltersLayer

from dffr_unet import dffr_unet_model, RepeatFiltersLayer

class dffr_unet_builder:
    @staticmethod
    def BuildDFFRUnet():
        '''
        Метод создания сети типа
        :return:
        '''
        inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size=BATCH_SIZE, name="input")

        (downsamplingLayer1, downsamplingResLayer1) = BuildDownSamplingBlock(inputs, 64, 64, (3, 3), 2, "ds_1")
        (downsamplingLayer2, downsamplingResLayer2) = BuildDownSamplingBlock(downsamplingLayer1, 64, 128, (3, 3), 2,
                                                                             "ds_2")
        (downsamplingLayer3, downsamplingResLayer3) = BuildDownSamplingBlock(downsamplingLayer2, 128, 256, (3, 3), 2,
                                                                             "ds_3")
        # (downsamplingLayer4, downsamplingResLayer4) = BuildDownSamplingBlock(downsamplingLayer3, 256, 512, (3, 3), 2, "ds_4")

        encodedConvLayer = BuildConvBlock(downsamplingLayer3, 1024, (3, 3), 1, "bneck_1")
        encodedConvLayer = BuildConvBlock(encodedConvLayer, 1024, (3, 3), 1, "bneck_2")

        # upsamplingLayer1 = BuildUpSamplingBlock(encodedConvLayer, downsamplingResLayer4, 512, (3, 3), 2, True, "us_1")
        upsamplingLayer2 = BuildUpSamplingBlock(encodedConvLayer, downsamplingResLayer3, 256, (3, 3), 2, True, "us_2")
        upsamplingLayer3 = BuildUpSamplingBlock(upsamplingLayer2, downsamplingResLayer2, 128, (3, 3), 2, True, "us_3")
        upsamplingLayer4 = BuildUpSamplingBlock(upsamplingLayer3, downsamplingResLayer1, 64, (3, 3), 2, True, "us_4")
        upsamplingLayer5 = BuildDoubleConvBlock(upsamplingLayer4, 4, (3, 3), True, "us_5")
        output = Softmax()(upsamplingLayer5)

        result = dffr_unet_model.DFFRUNet(inputs=inputs, outputs=output, name="dffr_unet")
        return result


def BuildDownSamplingBlock(currentLayer, currentFilters, endFilters, kernelSize, poolingSize = 1, baseName = "ds"):
    currentLayer = BuildConvBlock(currentLayer, currentFilters, kernelSize, 1, baseName = baseName + "_conv1")
    currentLayer = BuildConvBlock(currentLayer, endFilters, kernelSize, 1, baseName = baseName + "_conv2")
    pooledLayer = BuildConvBlock(currentLayer, endFilters, kernelSize, poolingSize, baseName = baseName + "_conv3")

    return (pooledLayer, currentLayer)


def BuildUpSamplingBlock(currentLayer, skipLayer, nFilters, kernelSize, poolingSize = 1, buildAttention = False, baseName = "us"):
    if buildAttention:
        gate = BuildGatingSignal(currentLayer, nFilters, kernelSize)
        attention = BuildAttentionBlock(skipLayer, gate, nFilters)

    currentLayer = Conv2DTranspose(
        filters=nFilters,
        kernel_size=kernelSize,
        strides=poolingSize,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-3),
        bias_regularizer=L2(l2=1e-3)
    ) (currentLayer)

    if buildAttention:
        currentLayer = Add() ([currentLayer, attention])
    else:
        currentLayer = Add() ([currentLayer, skipLayer])

    currentLayer = BuildDoubleConvBlock(currentLayer, nFilters, kernelSize, True, baseName = baseName + "_doubleconv")

    return currentLayer


def BuildDoubleConvBlock(currentLayer, nFilters, kernelSize, batchNorm=False, baseName = "doubleconv"):
    currentLayer = BuildConvBlock(currentLayer, nFilters, kernelSize, 1, baseName = baseName + "_conv1", batchNorm = batchNorm)
    currentLayer = BuildConvBlock(currentLayer, nFilters, kernelSize, 1, baseName = baseName + "_conv2", batchNorm = batchNorm)
    return currentLayer


def BuildConvBlock(currentLayer, nFilters, kernelSize, poolingSize, baseName, batchNorm = True):
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
        name=baseName + "_conv"
    )(currentLayer)

    if batchNorm:
        currentLayer = LeakyReLU(alpha=1e-02, name=baseName + "_relu")(currentLayer)
        currentLayer = BatchNormalization(epsilon=1.001e-05, name=baseName + "_bn")(currentLayer)

    return currentLayer


def BuildGatingSignal(layer, nFilters, kernelSize, batchNorm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gate feature map with the same dimension of the up layer feature map
    """
    layer = Conv2D(
        filters=nFilters,
        kernel_size=kernelSize,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-3),
        bias_regularizer=L2(l2=1e-3)
    ) (layer)
    if batchNorm:
        layer = BatchNormalization() (layer)

    layer = LeakyReLU(alpha = 1e-02) (layer)
    return layer


def BuildAttentionBlock(currentLayer, gate, nFilters):
    theta_x = Conv2D(nFilters, (2, 2), strides=(2, 2), padding='same')(currentLayer)  # 16

    phi_g = Conv2D(nFilters, (1, 1), padding='same')(gate)
    upsample_g = Conv2DTranspose(nFilters, (3, 3),
                                 strides=(1, 1),
                                 padding='same')(phi_g)  # 16

    concat_xg = Add() ([upsample_g, theta_x])
    act_xg = LeakyReLU(alpha = 1e-02) (concat_xg)
    psi = Conv2D(1, (1, 1), padding='same') (act_xg)
    sigmoid_psi = Activation('sigmoid') (psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_psi)  # 32

    upsample_psi = RepeatFiltersLayer.RepeatFiltersLayer(nFilters) (upsample_psi)

    y = Multiply() ([upsample_psi, currentLayer])

    result = Conv2D(nFilters, (1, 1), padding='same')(y)
    result_bn = BatchNormalization() (result)
    return result_bn

