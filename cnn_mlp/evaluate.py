import tensorflow as tf
import numpy as np
from PIL import Image
from data import GradingDataGenerator
from constants import RANDOM_SEED, BATCH_SIZE, IMAGE_SIZE, DATA_PATH, DATASET_PATH

import cnn_resnet as cnn_builder
#import cnn_imagenet as cnn_builder
#import cnn_attention as cnn_builder


def predict_labels(model):
    dataset = GradingDataGenerator.CreateGradingDataset(DATASET_PATH + "test",
                                                             DATA_PATH + "testImg2Labels.csv")

    dataset = \
        dataset. \
            batch(BATCH_SIZE, drop_remainder=True)

    predictions = model.predict(dataset)
    return predictions


def evaluate_metrics(model, y_true, y_pred):
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor(y_pred)

    y_true_labels = tf.math.argmax(y_true_tensor, axis = -1)
    y_pred_labels = tf.math.argmax(y_pred_tensor, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true_labels, y_pred_labels)

    model.compiled_metrics.update_state(y_true_tensor, y_pred_tensor)
    yBinary = tf.expand_dims(tf.cast(y_true_tensor[:, 0] == 0, tf.float32), axis=-1)
    yPredBinary = tf.expand_dims(tf.cast(y_pred_tensor[:, 0] < 0.5, tf.float32), axis=-1)
    model.binaryAccuracy.update_state(yBinary, yPredBinary)
    model.precision.update_state(yBinary, yPredBinary)
    model.recall.update_state(yBinary, yPredBinary)

    result = {m.name: m.result() for m in model.metrics}
    result["confusion_matrix"] = confusion_matrix
    return result


if __name__ == "__main__":
    #tf.config.run_functions_eagerly(True)

    model = cnn_builder.BuildCNNMLPModel()
    model.compile(metrics=[])

    predictions = predict_labels(model)

    metrics = evaluate_metrics(model, predictions[0], predictions[1])

    dummy = True