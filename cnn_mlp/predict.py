import tensorflow as tf
import numpy as np
from PIL import Image

#import cnn_resnet as cnn_builder
#import cnn_imagenet as cnn_builder
import cnn_attention as cnn_builder


def predict_label(imagePath: str):
    model = cnn_builder.BuildCNNMLPModel()
    model.compile(metrics = [])

    img = Image.open(imagePath)
    img.load()
    data = np.expand_dims(np.asarray(img, dtype="int32") / 255.0, axis = 0)
    predictions = model.predict(data, batch_size = 1)
    return predictions


if __name__ == "__main__":
    label = predict_label("D:\\Study\\Magister\\CourseworkProject\\main\\data\\eyepacs\\img_preprocessed\\train\\2. Moderate DR\\79_left.jpeg")
    dummy = True
