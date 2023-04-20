from keras.losses import Loss
from keras.utils import losses_utils
import tensorflow as tf
from . import PrecisionLoss, RecallLoss


class MulticlassF1(Loss):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None, nClasses: int = 2,
                 class_weights: [float] = None):
        super(MulticlassF1, self).__init__(reduction, name)

        self.precisionLoss = PrecisionLoss.MulticlassPrecision(reduction, name, nClasses, class_weights)
        self.recallLoss = RecallLoss.MulticlassRecall(reduction, name, nClasses, class_weights)
        self.simpleWeights = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0], dtype = tf.float32)

    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred, self.simpleWeights)

    def __call__(self, y_true, y_pred, sample_weight=None):
        print("Called Soft F1 loss which is expected to work with softmax output. Check your model configuration.")
        trueLabels = tf.cast(y_true, dtype = tf.float32)
        predictedLabels = tf.cast(y_pred, dtype=tf.float32)
        tp = tf.reduce_sum(trueLabels * predictedLabels, axis = 1)
        fp = tf.reduce_sum((1.0 - trueLabels) * predictedLabels, axis = 1)
        fn = tf.reduce_sum(trueLabels * (1.0 - predictedLabels), axis = 1)
        tp2 = tf.constant(2.0, dtype = tf.float32) * tp
        f1 = tp2 / (tp2 + fp + fn + 1.0e-10)
        f1_loss = tf.constant(1.0, dtype = tf.float32) - f1
        #loss = tf.cond(sample_weight == None, lambda : tf.reduce_sum(f1_loss, axis = 0), lambda : tf.reduce_sum(tf.cast(sample_weight, dtype = tf.float32) * f1_loss, axis = 0))
        loss = tf.reduce_sum(f1_loss, axis = 0)
        return loss
        #recallLoss = self.recallLoss.__call__(y_true, y_pred, sample_weight)
        #precisionLoss = self.precisionLoss.__call__(y_true, y_pred, sample_weight)
        #return 2 * recallLoss * precisionLoss / (recallLoss + precisionLoss)