import datetime

import streamlit as st
import numpy as np
import cv2
import pickle
import io
import os
import random
from fpdf import FPDF
from pypdf import PdfReader
from datetime import datetime
import PIL
from PIL import Image

import cnn_mlp.cnn_resnet as cnn_builder
import dffr_unet.dffr_unet_builder as unet_builder
import constants


@st.cache_resource
def BuildModels():
    classificationModel = cnn_builder.BuildCNNMLPModel()
    classificationModel.compile(metrics=[])

    segmentationModel = unet_builder.dffr_unet_builder.BuildDFFRUnet()
    segmentationModel.compile()

    return (classificationModel, segmentationModel)


def CreatePdfReport(patientCode, img, classificationModel, segmentationModel):
    imgToScore = np.asarray(img, dtype="float32") / 255.0
    imgToScore = 0.5 + 4 * (imgToScore - cv2.GaussianBlur(imgToScore, (0, 0), sigmaX=20))
    imgToScore = np.expand_dims(imgToScore, axis=0) * 255.0
    predictedLabels = classificationModel.predict(imgToScore, batch_size = 1)

    probability = np.max(predictedLabels)
    formattedProbability = f"{((1 - predictedLabels[0][0]) * 100.0):.2f}"
    label = np.argmax(predictedLabels)

    predictedSegmentationMaps = segmentationModel.predict(np.expand_dims(np.asarray(img, dtype = "float32"), axis = 0), batch_size=1)

    result = FPDF()
    result.add_page()

    result.add_font("DejaVu", "", "DejaVuSerif.ttf", uni = True)
    result.add_font("DejaVu", "B", "DejaVuSerif-Bold.ttf", uni=True)
    result.add_font("DejaVu", "I", "DejaVuSerif-BoldItalic.ttf", uni=True)

    result.set_font("DejaVu", "B", size = 18)
    result.cell(200, 10, txt = "Отчёт по проведённой диагностике", ln = 1, align = "C")

    result.ln(10)
    result.set_font("DejaVu", "", size=12)
    result.cell(200, 5, txt = f"Исследуемый пациент: {patientCode}", ln = 1,align = "L")
    result.cell(200, 5, txt = f"Дата и время обращения к системе: {datetime.now()}", ln = 1, align = "L")

    result.ln(10)
    result.set_font("DejaVu", "I", size=14)
    result.cell(200, 5, txt = "Системная оценка заболевания", ln = 1, align = "L")

    result.ln(5)
    result.set_font("DejaVu", "", size=12)
    if label == 0:
        result.cell(200, 5,
                    txt=f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %",
                    ln=1, align="L")
    else:
        result.cell(200, 5,
                    txt=f"Наиболее вероятная стадия заболевания - {label} ({(probability * 100.0):.3f} %).",
                    ln=1, align="L")
        result.cell(200, 5,
                    txt=f"Оцененная системой вероятность наличия заболевания: {formattedProbability} %",
                    ln=1, align="L")

    result.ln(8)
    result.set_font("DejaVu", "I", size=14)
    result.cell(200, 5, txt="Обнаруженные маркеры", ln=1, align="L")

    result.set_font("DejaVu", "", size=12)
    WritePilImageToPdfReport(result, "input.png", img,
                             15, 85, 64, 64)
    WritePilImageToPdfReport(result, "mask_rgb.png",
                             Image.fromarray((predictedSegmentationMaps[0, :, :, :3] * 255).astype("uint8"), mode = "RGB"),
                             90, 85, 64, 64)

    WritePilImageToPdfReport(result, "mask_ma_ha.png",
                             Image.fromarray((predictedSegmentationMaps[0, :, :, 0] * 255).astype("uint8"), mode="L"),
                             15, 160, 48, 48)

    WritePilImageToPdfReport(result, "mask_he_se.png",
                             Image.fromarray((predictedSegmentationMaps[0, :, :, 1] * 255).astype("uint8"), mode="L"),
                             75, 160, 48, 48)

    WritePilImageToPdfReport(result, "mask_od.png",
                             Image.fromarray((predictedSegmentationMaps[0, :, :, 2] * 255).astype("uint8"), mode="L"),
                             135, 160, 48, 48)

    return result


def WritePilImageToPdfReport(report, imgName, img : Image, x, y, w, h):
    img.save(imgName)
    img.close()
    report.image(imgName, x = x, y = y, w = w, h = h)
    os.remove(imgName)





(classifier, segmentator) = BuildModels()

selectedFunction = st.selectbox("Выбор системной функции",
                                ["Оценка стадии заболевания",
                                 "Сегментация маркеров заболевания",
                                 "Выгрузка отчёта"])
fileToProcessInfo = st.file_uploader("Загрузите снимков глазного дна", ["png", "jpg", "jpeg", "tif"])

if fileToProcessInfo is not None:
    processingCapture = Image.open(fileToProcessInfo).resize(constants.IMAGE_SIZE)
    processingCapture.load()

if fileToProcessInfo is not None and selectedFunction == "Оценка стадии заболевания":
    st.image(processingCapture, "Загруженный снимок")

    scoringCaptureArray = np.asarray(processingCapture, dtype="float32") / 255.0
    scoringCaptureArray = 0.5 + 4 * (scoringCaptureArray - cv2.GaussianBlur(scoringCaptureArray, (0, 0), sigmaX=20))
    scoringCaptureArray = np.expand_dims(scoringCaptureArray, axis=0) * 255.0
    predictedLabels = classifier.predict(scoringCaptureArray, batch_size=1)

    probability = np.max(predictedLabels)
    formattedProbability = f"{((1 - predictedLabels[0][0]) * 100.0):.2f}"
    label = np.argmax(predictedLabels)
    if label == 0:
        st.write(f"Вероятность наличия заболевания оценивается системой в {formattedProbability} %")
    else:
        st.write(f"Наиболее вероятная стадия заболевания - {label} ({(probability * 100.0):.3f} %). <br>",
                 f"Оцененная системой вероятность наличия заболевания: {formattedProbability} %",
                 unsafe_allow_html = True)


if fileToProcessInfo is not None and selectedFunction == "Сегментация маркеров заболевания":
    segmentingCaptureArray = np.expand_dims(np.asarray(processingCapture, dtype="float32"), axis = 0)
    predictedSegmentationMaps = segmentator.predict(segmentingCaptureArray, batch_size=1)

    column1, column2, column3 = st.columns(3)

    column1.image(processingCapture, "Загруженный снимок")
    column2.image(predictedSegmentationMaps[0, :, :, :3], "Общая сегментирующая маска")
    column3.image(predictedSegmentationMaps[0, :, :, 3], "Немаркированные области")

    column1.image(predictedSegmentationMaps[0, :, :, 0], "Сегментация участков нарушения работы кровеносной системы")
    column2.image(predictedSegmentationMaps[0, :, :, 1], "Сегментация участков с выделением жидкостей в сетчатку")
    column3.image(predictedSegmentationMaps[0, :, :, 2], "Сегментация диска зрительного нерва")

if fileToProcessInfo is not None and selectedFunction == "Выгрузка отчёта":
    patientCode = st.text_input("Код пациента", placeholder = "#000000")

    st.image(processingCapture, "Загруженный снимок")

    pdfReport = CreatePdfReport(patientCode, processingCapture, classifier, segmentator)
    reportBytes = pdfReport.output(name = "report.pdf", dest = "S").encode('latin-1')
    st.download_button(
        label = "Загрузить отчёт",
        data = reportBytes,
        file_name="report.pdf",
        mime="application/pdf")

