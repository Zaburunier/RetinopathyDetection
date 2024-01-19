import tensorflow as tf
import numpy as np
from PIL import Image

import constants
import dffr_unet_builder


def predict_label(imagePath: str):
    model = dffr_unet_builder.BuildDFFRUnet()
    model.compile(jit_compile = False, metrics=model.metrics)
    model.save("dffr.keras")

    img = Image.open(imagePath)
    img.load()
    img = img.resize((constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1]))
    data = np.expand_dims(np.asarray(img, dtype="int32") / 1.0, axis = 0)
    predictions = model.predict(data, batch_size = 1)
    return predictions


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    segmentation = predict_label("D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\train\\img\\IDRiD_22.jpg")
    segmentation_opaque = segmentation[:, :, :, :3]
    segmentation_ma_ha = segmentation[:, :, :, 0]
    segmentation_he_se = segmentation[:, :, :, 1]
    segmentation_od = segmentation[:, :, :, 2]
    segmentation_us = segmentation[:, :, :, 3]

    dummy = True