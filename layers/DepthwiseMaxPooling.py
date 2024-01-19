import tensorflow as tf

class DepthwiseMaxPooling(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(DepthwiseMaxPooling, self).__init__(*args, **kwargs)


    def compute_output_shape(self, inputShape):
        inputShape = tf.TensorShape(inputShape).as_list()
        return tf.TensorShape([inputShape[0], inputShape[1], inputShape[2], 1])


    def call(self, inputs):
        return tf.reduce_max(inputs, axis = [3], keepdims=True)