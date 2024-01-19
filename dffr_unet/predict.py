import tensorflow as tf
import numpy as np
from PIL import Image

import constants
import dffr_unet_builder


def predict_label(imagePath: str):
    model = dffr_unet_builder.BuildDFFRUnet()
    model.compile(jit_compile = False, metrics=model.metrics)

    img = Image.open(imagePath)
    img.load()
    img = img.resize((constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1]))
    data = np.expand_dims(np.asarray(img, dtype="int32") / 1.0, axis = 0)
    predictions = model.predict(data, batch_size = 1)
    return predictions


if __name__ == "__main__":
    label = predict_label("D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\img\\IDRiD_62.jpg")
    dummy = True