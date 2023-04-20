from tensorflow.python.keras.metrics import Metric
from sklearn.metrics import recall_score
import numpy as np
import tensorflow as tf

import constants


class WeightedRecall(Metric):

    def __init__(self, name: str = "recall", dtype = None, averagingMode : str = None, class_labels: [str] = None):
        if averagingMode is not None:
            name += "_" + averagingMode

        super(WeightedRecall, self).__init__(name, dtype)

        self.avgMode = averagingMode
        self.trueLabels = tf.constant(0.0, shape=[0, 5])
        self.predictedLabels = tf.constant(0.0, shape=[0, 5])

    def reset_state(self):
        self.trueLabels = tf.constant(0.0, shape = [0, 5])
        self.predictedLabels = tf.constant(0.0, shape = [0, 5])


    def call(self, inputs, *args, **kwargs):
        y_true, y_pred = inputs
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (tf.constant(1.0, shape = y_pred) - y_pred), tf.float32), axis=0) + 1.0e-10
        return tp / (tp + fn)


    def update_state(self, y_true, y_pred, *args, **kwargs):
        #newTrue = tf.constant(y_true, shape = y_true.shape)
        #newPredicted = tf.constant(y_pred, shape = y_true.shape)
        self.trueLabels = tf.concat([self.trueLabels, y_true], 0)
        self.predictedLabels = tf.concat([self.predictedLabels, y_pred], 0)


    def result(self):
        tp = tf.reduce_sum(tf.cast(self.trueLabels * self.predictedLabels, tf.float32), axis = 0)
        fn = tf.reduce_sum(tf.cast(self.trueLabels * (tf.constant(1.0, shape = self.predictedLabels.shape) - self.predictedLabels), tf.float32), axis = 0) + 1.0e-10
        #zeroMask = tp + fn < 1.0e-10

        result = tp / (tp + fn)
        #result[zeroMask] = tf.constant(0.0)
        #print(result.numpy())
        #print(result.shape)
        return {"Recall-0": result[0],
                "Recall-1": result[1],
                "Recall-2": result[2],
                "Recall-3": result[3],
                "Recall-4": result[4]}



