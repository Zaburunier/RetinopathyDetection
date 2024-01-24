import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

import constants
from dffr_unet import dffr_unet_builder


def predict_label(imagePath: str):
    model = dffr_unet_builder.dffr_unet_builder.BuildDFFRUnet()
    model.compile(jit_compile = False, metrics=model.metrics)

    img = get_image(imagePath)
    data = np.expand_dims(img, axis = 0)
    predictions = model.predict(data, batch_size = 1)

    img /= 255.0
    blurredImg = cv2.GaussianBlur(img, (0, 0), sigmaX=20)
    sharpenedImg = 4 * (img - blurredImg)
    preprocessedImg = 0.5 + sharpenedImg

    return predictions


def get_groundtruth(basePath: str, imgName : str):
    imgs = []

    imgMa = get_image(basePath + "ma\\" + imgName + "_MA.tif")
    imgs.append(imgMa)

    imgHa = get_image(basePath + "ha\\" + imgName + "_HE.tif")
    imgs.append(imgHa)

    imgHe = get_image(basePath + "he\\" + imgName + "_EX.tif")
    imgs.append(imgHe)

    imgSe = get_image(basePath + "se\\" + imgName + "_SE.tif")
    imgs.append(imgSe)

    imgOd = get_image(basePath + "od\\" + imgName + "_OD.tif")
    imgs.append(imgOd)

    imgMaHa = imgMa + imgHa
    imgs.append(imgMaHa)

    imgHeSe = imgHe + imgSe
    imgs.append(imgHeSe)

    imgColored = np.stack([np.any(imgMaHa, axis = -1), np.any(imgHeSe, axis = -1), np.any(imgOd, axis = -1)], axis = -1)
    imgs.append(imgColored)

    return imgs


def get_image(imagePath: str):
    img = Image.open(imagePath)
    img.load()
    img = img.resize((constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1]))
    return np.asarray(img, dtype = "float32")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    segmentationLabels = get_groundtruth("D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\train\\mask\\", "IDRiD_22")

    segmentation = predict_label("D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\train\\img\\IDRiD_22.jpg")
    segmentation_opaque = segmentation[:, :, :, :3]
    segmentation_ma_ha = segmentation[:, :, :, 0]
    segmentation_he_se = segmentation[:, :, :, 1]
    segmentation_od = segmentation[:, :, :, 2]
    segmentation_us = segmentation[:, :, :, 3]

    dummy = True