import tensorflow as tf
import numpy as np
from PIL import Image

import constants
import mtunet_builder


def predict_label(imagePath: str):
    model = mtunet_builder.BuildMTUnet()
    model.compile(jit_compile = False, metrics=model.metrics)

    img = Image.open(imagePath)
    img.load()
    img = img.resize((constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1]))
    data = np.expand_dims(np.asarray(img, dtype="int32") / 255.0, axis = 0)
    predictions = model.predict(data, batch_size = 1)
    return predictions


if __name__ == "__main__":
    label = predict_label("D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\train\\img\\IDRiD_02.jpg")
    dummy = True