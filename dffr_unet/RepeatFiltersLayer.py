import tensorflow as tf

class RepeatFiltersLayer(tf.keras.layers.Layer):
    def __init__(self, nRepeats):
        super(RepeatFiltersLayer, self).__init__()

        self.tileRatio = tf.constant([1, 1, 1, nRepeats], dtype = tf.int32)


    def compute_output_shape(self, inputShape):
        inputShape = tf.TensorShape(inputShape).as_list()
        return tf.TensorShape(inputShape * self.tileRatio)


    def call(self, inputs):
        return tf.tile(inputs, self.tileRatio)