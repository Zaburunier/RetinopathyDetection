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

    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        recallLoss = -self.recallLoss.__call__(y_true, y_pred, sample_weight)
        precisionLoss = -self.precisionLoss.__call__(y_true, y_pred, sample_weight)
        return -2 * recallLoss * precisionLoss / (recallLoss + precisionLoss)