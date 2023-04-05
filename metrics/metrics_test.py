import tensorflow as tf
import WeightedRecallMetric, WeightedPrecision

def main():
    trueLabels = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    predictedLabels = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
    metric = WeightedRecall.WeightedRecall()
    metric.update_state(trueLabels, predictedLabels)
    print(metric.result())


if __name__=="__main__":
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    main()