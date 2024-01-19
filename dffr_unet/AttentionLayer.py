import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()


    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)

        super(AttentionLayer, self).build(input_shape)


    def call(self, inputs):
        scores = tf.matmul(inputs, self.W) + self.b
        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = attention_weights * inputs
        output = tf.reduce_sum(context_vector, axis=1)

        return output