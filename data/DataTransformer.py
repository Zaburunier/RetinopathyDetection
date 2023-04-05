import PIL.Image
from PIL import Image
import os, os.path
import numpy as np
from pandas import read_csv, DataFrame

class DataTransformer:
    '''
    Класс для предварительной подготовки данных к работе с нейронной сетью.
    У нас изображения прямоугольные и большого размера. Мы хотим корректно привести их к квадрату и сжать
    '''


    @staticmethod
    def PrepareData(sourceFolder: str, destinationFolder: str, targetImageSize: int = 256):
        '''
        Метод преобразования изображений в удобный для поглощения нейронками формат
        :param sourceFolder: Расположение исходных данных
        :param destinationFolder: Расположение подготовленных данных
        :param targetImageSize: Размер изображений (они будут квадратными)
        '''
        print("Transform data method entered")
        validExtensions = [".jpg", ".jpeg", ".png"]
        for filename in os.listdir(sourceFolder):
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            print("Working with ", filename, "...", sep = "")
            sourceImage = Image.open(os.path.join(sourceFolder, filename))
            transformedImage = DataTransformer.TransformImage(sourceImage, targetImageSize)
            transformedImage.save(os.path.join(destinationFolder, filename))
            print(filename, " transform finished.", sep = "")

        print("Transform data method exited")


    @staticmethod
    def TransformImage(sourceImage : Image, targetImageSize: int = 256) -> Image:
        '''
        Метод преобразования изображения
        :param sourceImage: Исходное изображение
        :param targetImageSize: Размер изображения (оно будет квадратным)
        :return: Преобразованное изображение
        '''
        # Преобразование представляет собой два действия:
        # 1. Приведение изображения к квадратному формату
        # 2. Сжатие квадрата до требуемого размера

        # С приведением к квадрату могут быть сложности, поскольку есть изображения, где глаз попал не целиком
        # Действуем так:
        # 1. Если глаз попал целиком, то просто обрезаем бока;
        # 2. Если глаз попал не целиком, то добавляем сверху и снизу чёрные строки пикселей.
        # Определить попадание целого глаза можно по наличию сверху и снизу в исходном изображении чёрных пикселей
        sourceImageArray = np.asarray(sourceImage).astype(np.uint8)

        # Ищем чёрные пиксели по строкам и по столбцам (так выделим границы глаза)
        mask = sourceImageArray <= 1e-03
        blackPixels = np.all(mask, axis = 2)

        blackRows = np.all(blackPixels, axis = 1)
        blackColumns = np.all(blackPixels, axis = 0)

        # Выделяем границы глаза на исходном изображении
        nonBlackIndices = np.nonzero(np.logical_not(blackRows))
        rowMin = nonBlackIndices[0][0]
        rowMax = nonBlackIndices[0][-1]
        nonBlackIndices = np.nonzero(np.logical_not(blackColumns))
        columnMin = nonBlackIndices[0][0]
        columnMax = nonBlackIndices[0][-1]

        rowDelta = rowMax - rowMin
        columnDelta = columnMax - columnMin
        delta = rowDelta - columnDelta
        if delta > 0:
            # Глаз обрезан сверху-снизу, добавляем чёрные строки
            destinationImageArray = np.zeros([rowDelta + 1, rowDelta + 1, 3], dtype = np.uint8)
            destinationImageArray[:, (delta // 2):((delta // 2) + columnDelta + 1), :] = \
                sourceImageArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]
        elif delta < 0:
            # Глаз обрезан слева-справа, добавляем чёрные колонки
            destinationImageArray = np.zeros([columnDelta + 1, columnDelta + 1, 3], dtype = np.uint8)
            destinationImageArray[(-delta // 2):((-delta // 2) + rowDelta + 1), :, :] = \
                sourceImageArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]
        else: # delta == 0
            # Глаз занял ровно квадратную область, ничего не добавляем
            destinationImageArray = sourceImageArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]

        return PIL.Image.fromarray(destinationImageArray, mode = "RGB").resize((targetImageSize, targetImageSize))


    @staticmethod
    def OrganizeImages(sourceFolder: str, label_csv_filename : str):
        '''
        Создание структуры папок для организованного доступа к датасету
        :param sourceFolder: Расположение исходных файлов
        :return:
        '''

        labelFile = read_csv(label_csv_filename)
        labels = labelFile.to_numpy()[:, 1]
        classSubfolders = ["0. No DR/", "1. Mild DR/", "2. Moderate DR/", "3. Severe DR/", "4. Profilerative DR/"]
        if not os.path.exists(sourceFolder + "test/"):
            os.makedirs(sourceFolder + "test/")

        if not os.path.exists(sourceFolder + "train/"):
            os.makedirs(sourceFolder + "train/")

        for i in range(5):
            if not os.path.exists(sourceFolder + "test/" + classSubfolders[i]):
                os.makedirs(sourceFolder + "test/" + classSubfolders[i])

            if not os.path.exists(sourceFolder + "train/" + classSubfolders[i]):
                os.makedirs(sourceFolder + "train/" + classSubfolders[i])


        print("Organize data method entered")
        validExtensions = [".jpg", ".jpeg", ".png"]
        imageFilenames = []
        for filename in os.listdir(sourceFolder):
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            imageFilenames.append(filename)

        # 1. Разбиваем изображения на тренировочные и проверочные
        # 2. Раскладываем изображения по подпапкам, соответствующим классу заболевания
        length = len(imageFilenames)
        indices = np.arange(length)
        testIndices = np.random.choice(indices, int(0.1 * float(length)), replace = False)
        indices = np.delete(indices, testIndices)

        for index in testIndices:
            print("Working with ", imageFilenames[index], "...", sep="")
            image = PIL.Image.open(sourceFolder + imageFilenames[index])
            dummy = sourceFolder + "test/" + classSubfolders[labels[index]] + imageFilenames[index]
            image.save(dummy)
            print(imageFilenames[index], " organize finished.", sep = "")

        for index in indices:
            print("Working with ", imageFilenames[index], "...", sep="")
            image = PIL.Image.open(sourceFolder + imageFilenames[index])
            dummy = sourceFolder + "train/" + classSubfolders[labels[index]] + imageFilenames[index]
            image.save(dummy)
            print(imageFilenames[index], " organize finished.", sep = "")
