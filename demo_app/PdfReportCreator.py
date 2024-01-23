import numpy as np
import cv2
import os
from fpdf import FPDF
from datetime import datetime
from PIL import Image


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