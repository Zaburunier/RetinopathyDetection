from constants import IMAGE_SIZE, BATCH_SIZE

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalMaxPool2D, Conv2DTranspose, LeakyReLU, Add, Concatenate
from tensorflow.keras.layers import Input, Rescaling, RandomCrop, RandomFlip
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Zeros, HeNormal, HeUniform
from tensorflow.keras.regularizers import L2

import mtunet


def BuildMTUnet():
    '''
    Метод создания сети типа MTUNet (одна из базовых моделей)
    :return:
    '''
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size = BATCH_SIZE, name="input")
    #flip = RandomFlip() (inputs)

    #vgg = VGG16(
    #    include_top = False,
    #    weights = 'imagenet',
    #    input_tensor = None,
    #    inputShape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    #    pooling = None,
    #) (flip)

    # Кодировщик (архитектура полностью копирует VGG-16)

    conv_11 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="swish",
            name="conv_11",
            kernel_initializer = HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (inputs)

    conv_12 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_12",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (conv_11)

    mp_21 = MaxPool2D(pool_size = (2, 2), name = "mp_21") (conv_12)

    conv_21 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_21",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (mp_21)

    conv_22 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_22",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (conv_21)

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
        bias_regularizer = L2(l2=2e-03)
    ) (mp_31)

    conv_32 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_32",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (conv_31)

    conv_33 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="conv_33",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
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
        bias_regularizer = L2(l2=2e-03)
    )(conv_52)

    # Ответвление под оценку степени тяжести
    gmp = GlobalMaxPool2D(name = "gmp") (conv_53)
    grading_output = Dense(5,
                           activation = "softmax",
                           kernel_initializer=HeNormal(),
                           kernel_regularizer=L2(l2=1e-03),
                           bias_regularizer = L2(l2=2e-03),
                           name = "grade_predict") (gmp)

    # Ответвление под сегментацию (зеркальное отражение VGG)
    deconv_53 = Conv2DTranspose(
        filters = 256,
        kernel_size = (3, 3),
        strides = 1,
        padding="same",
        activation="swish",
        name = "deconv_53",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (conv_53)

    concat_53 = Concatenate() ([deconv_53, conv_53])

    deconv_52 = Conv2DTranspose(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_52",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    ) (concat_53)

    deconv_51 = Conv2DTranspose(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_51",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_52)

    deconv_43 = Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="swish",
        name="deconv_43",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_51)

    concat_43 = Concatenate()([deconv_43, conv_43])

    deconv_42 = Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_42",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(concat_43)

    deconv_41 = Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_41",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_42)

    deconv_33 = Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="swish",
        name="deconv_33",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_41)

    concat_33 = Concatenate()([deconv_33, conv_33])

    deconv_32 = Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_32",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(concat_33)

    deconv_31 = Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_31",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_32)

    deconv_23 = Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="swish",
        name="deconv_23",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_31)

    concat_23 = Concatenate()([deconv_23, conv_22])

    deconv_22 = Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_22",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(concat_23)

    deconv_21 = Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_21",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_22)

    deconv_13 = Conv2DTranspose(
        filters=16,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation="swish",
        name="deconv_13",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_21)

    concat_13 = Concatenate()([deconv_13, conv_12])

    deconv_12 = Conv2DTranspose(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_12",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(concat_13)

    deconv_11 = Conv2DTranspose(
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="swish",
        name="deconv_11",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_12)

    deconv_output = Conv2DTranspose(
        filters=4,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="softmax",
        name="segm_predict",
        kernel_initializer=HeNormal(),
        kernel_regularizer=L2(l2=1e-03),
        bias_regularizer = L2(l2=2e-03)
    )(deconv_11)

    result = mtunet.MTUnet(inputs = inputs, outputs = [grading_output, deconv_output], name = "mtunet")
    return result