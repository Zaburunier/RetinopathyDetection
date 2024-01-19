import tensorflow as tf

class MatmulLayer(tf.keras.layers.Layer):
    def __init__(self, nFilters, kernelSize):
        super(MatmulLayer, self).__init__()


    def build(self, input_shape):
        super(MatmulLayer, self).build(input_shape)


    def call(self, inputs):
        return tf.linalg.matmul(inputs[0], inputs[1])