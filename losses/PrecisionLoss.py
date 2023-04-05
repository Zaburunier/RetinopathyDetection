from keras.losses import Loss
from keras.utils import losses_utils
import tensorflow as tf


class MulticlassPrecision(Loss):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None, nClasses: int = 2,
                 class_weights: [float] = None):
        super(MulticlassPrecision, self).__init__(reduction, name)

        self.nClasses = nClasses
        if class_weights is None:
            class_weights = tf.constant(1 / nClasses, shape=(1, nClasses))

        self.class_weights = tf.reshape(tf.convert_to_tensor(class_weights, dtype=tf.float32), shape=(1, nClasses))

    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # trueLabels = self.class_weights * y_true
        predictedLabels = tf.nn.softmax(y_pred, axis=1)  # self.class_weights * tf.nn.softmax(y_pred, axis = 1)
        # predictedLabels = tf.nn.softmax(predictedLabels, axis = 1)
        tp = tf.reduce_sum(self.class_weights * tf.cast(y_true * predictedLabels, tf.float32), axis=1) + 1.0e-10
        fp = tf.reduce_sum(self.class_weights * tf.cast(((tf.constant(1.0, shape=(1, self.nClasses)) - y_true) * predictedLabels), tf.float32), axis=1) + 1.0e-10
        result = tp / (tp + fp)
        batchResult = tf.reduce_sum(result, axis=0) / tp.shape[0]
        return -batchResult