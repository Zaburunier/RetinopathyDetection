import datetime
import math
import os
import time

import streamlit as st
import extra_streamlit_components as stx
from streamlit_cropper import st_cropper
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image
from PdfReportCreator import CreatePdfReport

import requests
from urllib.parse import urlencode

import cnn_mlp.cnn_resnet as cnn_builder
import dffr_unet.dffr_unet_builder as unet_builder
import constants




def LoadWeightsFromYaDisk(url, filename):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = url

    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Загружаем файл и сохраняем его
    download_response = requests.get(download_url)
    with open(filename, 'wb') as f:  # Здесь укажите нужный путь к файлу
        f.write(download_response.content)


@st.cache_resource
def BuildModels():
    classificationModel = cnn_builder.BuildCNNMLPModel()
    classificationModel.compile(metrics=[])

    LoadWeightsFromYaDisk("https://disk.yandex.ru/d/eLJdaT711pJ3pQ", "./weights/classifier.h5")
    classificationModel.load_weights("./weights/classififer.h5")

    segmentationModel = unet_builder.dffr_unet_builder.BuildDFFRUnet()
    segmentationModel.compile()

    LoadWeightsFromYaDisk("https://disk.yandex.ru/d/yQw4NmO_LPZHJA", "./weights/segmentator.h5")
    segmentationModel.load_weights("./weights/segmentator.h5")

    return (classificationModel, segmentationModel)


def CropImg(img : Image) -> Image:
    return st_cropper(img, box_color="#00FF00", aspect_ratio=(1, 1)).resize(constants.IMAGE_SIZE)


def DrawImageScoring(img : Image, container = None):
    imgArray = np.asarray(img, dtype="float32") / 255.0
    imgArray = 0.5 + 4 * (imgArray - cv2.GaussianBlur(imgArray, (0, 0), sigmaX=20))
    imgArray = np.expand_dims(imgArray, axis=0) * 255.0
    predictedLabels = classifier.predict(imgArray, batch_size=1)

    probability = np.max(predictedLabels)
    formattedProbability = f"{((1 - predictedLabels[0][0]) * 100.0):.2f}"
    label = np.argmax(predictedLabels)
    if label == 0:
        if container is None:
            if probability < 0.2:
                st.success(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
            else:
                st.warning(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
        else:
            if probability < 0.2:
                container.success(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
            else:
                container.warning(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
    else:
        if container is None:
            st.error(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
            st.info(f"Наиболее вероятная стадия заболевания - {label} ({(probability * 100.0):.3f} %)")
        else:
            container.error(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
            container.info(f"Наиболее вероятная стадия заболевания - {label} ({(probability * 100.0):.3f} %)")

    transparentImg = np.asarray(img, dtype="float32") / 255.0
    transparentImgMask = np.all(transparentImg < 1e-01, axis=-1, keepdims=True)
    transparentImg = np.concatenate([transparentImg, 1.0 - transparentImgMask.astype(float)], axis=-1)


def toggle_animation():
    st.session_state.playLoopedMaskAnimation = not st.session_state.playLoopedMaskAnimation
    #st.session_state.startTime = time.time_ns()


def DrawImageSegmentation(img : Image, container = None):
    imgArray = np.asarray(img, dtype="float32")
    predictedSegmentationMaps = segmentator.predict(np.expand_dims(imgArray, axis=0), batch_size=1)

    if container is None:
        column1, column2, column3 = st.columns(3)
    else:
        column1, column2, column3 = container.columns(3)

    if not st.session_state.playLoopedMaskAnimation:
        column1.image(predictedSegmentationMaps[0, :, :, 0], "Нарушения кровеносной системы", use_column_width = "auto")
        column2.image(predictedSegmentationMaps[0, :, :, 1], "Выделение жидкостей в сетчатку", use_column_width = "auto")
        column3.image(predictedSegmentationMaps[0, :, :, 2], "Диск зрительного нерва", use_column_width = "auto")
    else:
        bloodMarkersPlaceholder = column1.empty()
        exudatesPlaceholder = column2.empty()
        opticalDiskPlaceholder = column3.empty()

        nFrames = 50
        timeInterval = .5 / nFrames
        imgToInterpolate = imgArray / 255.0
        bloodMarkersMaskToInterpolate = np.expand_dims(predictedSegmentationMaps[0, :, :, 0], axis = -1)
        exudatesMaskToInterpolate = np.expand_dims(predictedSegmentationMaps[0, :, :, 1], axis = -1)
        opticalDiskMaskToInterpolate = np.expand_dims(predictedSegmentationMaps[0, :, :, 2], axis = -1)

        bloodMarkersMaskedImgToInterpolate = imgToInterpolate * bloodMarkersMaskToInterpolate
        exudatesMaskedImgToInterpolate = imgToInterpolate * exudatesMaskToInterpolate
        opticalDiskMaskedImgToInterpolate = imgToInterpolate * opticalDiskMaskToInterpolate

        sinRatio = 0.375 * math.pi * 1.0e-09
        while True:
            t = 0.5 * (1 + math.sin(sinRatio * (time.time_ns() - st.session_state.startTime)))

            bloodMarkersPlaceholder.image(
                ((1 - t) ** 2) * bloodMarkersMaskToInterpolate +
                2 * t * (1 - t) * bloodMarkersMaskedImgToInterpolate +
                t * t * imgToInterpolate,
                "Нарушения кровеносной системы", use_column_width = "auto")

            exudatesPlaceholder.image(
                ((1 - t) ** 2) * exudatesMaskToInterpolate +
                2 * t * (1 - t) * exudatesMaskedImgToInterpolate +
                t * t * imgToInterpolate,
                "Выделение жидкостей в сетчатку", use_column_width = "auto")

            opticalDiskPlaceholder.image(
                ((1 - t) ** 2) * opticalDiskMaskToInterpolate +
                2 * t * (1 - t) * opticalDiskMaskedImgToInterpolate +
                t * t * imgToInterpolate,
                "Диск зрительного нерва", use_column_width = "auto")

            time.sleep(timeInterval)


def HandleClassification(selectedImgs, selectedTabId):
    st.image(selectedImgs[selectedTabId], "Загруженный снимок")

    col1, col2 = st.columns(2)

    with col1:
        selectedImgs[selectedTabId] = CropImg(selectedImgs[selectedTabId])
    col2.image(selectedImgs[selectedTabId], "Выделенная область")

    preprocessedSelectedImage = segmentator.PrepareImage(
        np.asarray(selectedImgs[selectedTabId]) / 255.0).numpy()
    tempMin = np.min(preprocessedSelectedImage)
    tempMax = np.max(preprocessedSelectedImage)

    DrawImageScoring(selectedImgs[selectedTabId].resize(constants.IMAGE_SIZE))

    st.divider()
    image_comparison(selectedImgs[selectedTabId].copy(),
                     (255.0 * (preprocessedSelectedImage - tempMin) / (tempMax - tempMin)).astype("uint8"),
                     label1="До предобработки",
                     label2="После предобработки",
                     width=2 * constants.IMAGE_SIZE[0])


def HandleSegmentation(selectedImgs, selectedTabId):
    if "playLoopedMaskAnimation" not in st.session_state:
        st.session_state.playLoopedMaskAnimation = True
        st.session_state.startTime = time.time_ns()

    st.image(selectedImgs[selectedTabId], "Загруженный снимок")

    col1, col2 = st.columns(2)

    with col1:
        selectedImgs[selectedTabId] = CropImg(selectedImgs[selectedTabId])
    col2.image(selectedImgs[selectedTabId], "Выделенная область")

    preprocessedSelectedImage = segmentator.PrepareImage(
        np.asarray(selectedImgs[selectedTabId]) / 255.0).numpy()
    tempMin = np.min(preprocessedSelectedImage)
    tempMax = np.max(preprocessedSelectedImage)

    st.session_state.playLoopedMaskAnimation = st.toggle("Анимация наложения масок сегментации")
    resultContainer = st.container()

    st.divider()
    image_comparison(selectedImgs[selectedTabId].copy(),
                     (255.0 * (preprocessedSelectedImage - tempMin) / (tempMax - tempMin)).astype("uint8"),
                     label1="До предобработки",
                     label2="После предобработки",
                     width=2 * constants.IMAGE_SIZE[0])
    st.divider()

    DrawImageSegmentation(selectedImgs[selectedTabId].resize(constants.IMAGE_SIZE), resultContainer)



def HandleOptions(selectedImgs, selectedTabId, selectedFunction):
    if len(selectedImgs) > 0 and selectedFunction == "Оценка стадии заболевания":
        if selectedTabId != "None":
            selectedTabId = int(selectedTabId)

            HandleClassification(selectedImgs, selectedTabId)

    if len(selectedImgs) > 0 and selectedFunction == "Сегментация маркеров заболевания":
        if selectedTabId != "None":
            selectedTabId = int(selectedTabId)

            HandleSegmentation(selectedImgs, selectedTabId)

    if len(selectedImgs) > 0 and selectedFunction == "Выгрузка отчёта":
        if selectedTabId != "None":
            selectedTabId = int(selectedTabId)

            if "patientCode" not in st.session_state:
                st.session_state.patientCode = "#000000"

            st.session_state.patientCode = st.text_input("Код пациента", value=st.session_state.patientCode)

            st.image(selectedImgs[selectedTabId], "Загруженный снимок")

            col1, col2 = st.columns(2)

            with col1:
                selectedImgs[selectedTabId] = CropImg(selectedImgs[selectedTabId])
            col2.image(selectedImgs[selectedTabId], "Выделенная область")

            pdfReport = CreatePdfReport(st.session_state.patientCode,
                                        selectedImgs[selectedTabId].resize(constants.IMAGE_SIZE), classifier,
                                        segmentator)
            reportBytes = pdfReport.output(name="report.pdf", dest="S").encode('latin-1')
            st.download_button(
                label="Загрузить отчёт",
                data=reportBytes,
                file_name=f"ДиагОтчёт-{st.session_state.patientCode}-{datetime.datetime.now()}.pdf",
                mime="application/pdf")

            preprocessedSelectedImage = segmentator.PrepareImage(
                np.asarray(selectedImgs[selectedTabId]) / 255.0).numpy()
            tempMin = np.min(preprocessedSelectedImage)
            tempMax = np.max(preprocessedSelectedImage)

            image_comparison(selectedImgs[selectedTabId].copy(),
                             (255.0 * (preprocessedSelectedImage - tempMin) / (tempMax - tempMin)).astype("uint8"),
                             label1="До предобработки",
                             label2="После предобработки",
                             width=2 * constants.IMAGE_SIZE[0])



(classifier, segmentator) = BuildModels()

st.session_state.lastReloadTime = time.time_ns()

st.title("Проект 1491")

loadOption = st.radio("Выберите опцию загрузки", ["Указание файла", "Указание папки с набором файлов"])
selectedImgFilenames = []
selectedImgs = []
if loadOption == "Указание файла":
    loadFullFolder = False

    fileToProcessInfo = st.file_uploader("Загрузите снимок глазного дна", ["png", "jpg", "jpeg", "tif"])
    if fileToProcessInfo is not None:
        img = Image.open(fileToProcessInfo).resize(constants.IMAGE_SIZE)
        img.load()
        selectedImgs.append(img)
        selectedImgFilenames.append(fileToProcessInfo.name)

elif loadOption == "Указание папки с набором файлов":
    loadFullFolder = True

    loadPath = st.text_input("Укажите папку для загрузки")
    maxToLoad = st.number_input("Максимальное количество файлов", value = 0)

    counter = maxToLoad
    if os.path.exists(loadPath):
        for file in os.scandir(loadPath):
            if (os.path.splitext(file.name)[1] in [".png", ".jpg", ".jpeg", ".tif"]):
                img = Image.open(os.path.join(loadPath, file)).resize(constants.IMAGE_SIZE)
                img.load()
                selectedImgs.append(img)
                selectedImgFilenames.append(file.name)

            if (counter == 1):
                break

            counter -= 1

selectedFunction = st.selectbox("Выбор системной функции",
                                ["Оценка стадии заболевания",
                                 "Сегментация маркеров заболевания",
                                 "Выгрузка отчёта"])

tabs = []
for i in range(len(selectedImgs)):
    tabs.append(stx.TabBarItemData(id=i, title = selectedImgFilenames[i], description = ""))

selectedTabId = stx.tab_bar(tabs)

HandleOptions(selectedImgs, selectedTabId, selectedFunction)

