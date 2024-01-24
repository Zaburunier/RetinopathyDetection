import PIL.Image
import pandas as pd
from PIL import Image
import os, os.path
import numpy as np
import cv2
from pandas import read_csv
import multiprocessing
import constants

class DataTransformer:
    '''
    Класс для предварительной подготовки данных к работе с нейронной сетью.
    У нас изображения прямоугольные и большого размера. Мы хотим корректно привести их к квадрату и сжать
    '''


    @staticmethod
    def PrepareDataOnSingleThread(sourceFolder: str, destinationFolder: str, targetImageSize: int = 256):
        '''
        Метод преобразования изображений в удобный для поглощения нейронками формат
        :param sourceFolder: Расположения исходных данных
        :param destinationFolder: Расположение подготовленных данных
        :param targetImageSize: Размер изображений (они будут квадратными)
        '''
        print("Transform data method entered")
        validExtensions = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
        # Старая версия с одной папкой
        for filename in os.listdir(sourceFolder):
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            print("Working with ", filename, "...", sep="")
            sourceImage = Image.open(os.path.join(sourceFolder, filename))
            transformedImage = DataTransformer.TransformImage(sourceImage, targetImageSize)
            print(filename, " transform finished. ", transformedImage, sep="")
            if (transformedImage is not None):
                transformedImage.save(os.path.join(destinationFolder, filename))

        print("Transform data method exited")


    @staticmethod
    def PrepareData(sourceFolder: str, destinationFolder: str, targetImageSize: int = 256, nThreads : int = 1):
        #if nThreads == 1:
        #    DataTransformer.PrepareDataOnSingleThread(sourceFolder, destinationFolder, targetImageSize)
        #    return

        print("Transform data method entered")
        validExtensions = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
        filenames = []
        for filename in os.listdir(sourceFolder):
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            filenames.append(filename)

        nFiles = len(filenames)
        processes = []
        for i in range(nThreads):
            minIndex = int((i / nThreads) * nFiles)
            maxIndex = int(((i + 1) / nThreads) * nFiles)
            proc = multiprocessing.Process(target = DataTransformer.PrepareImages, args = (sourceFolder, filenames[minIndex:maxIndex], destinationFolder, targetImageSize))
            proc.start()
            processes.append(proc)

        for i in range(nThreads):
            processes[i].join()

        print("Transform data method exited")


    @staticmethod
    def PrepareImages(sourceFolder: str, filenames: [str], destinationFolder: str, targetImageSize: int = 256):
        validExtensions = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
        for filename in filenames:
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            print("Working with ", filename, "...", sep="")
            sourceImage = Image.open(os.path.join(sourceFolder, filename))
            #transformedImage = DataTransformer.TransformImage(sourceImage, targetImageSize)
            #if (transformedImage is not None):
            #    transformedImage.save(os.path.join(destinationFolder, filename))
            transformedImage = DataTransformer.TransformImage2(sourceImage, targetImageSize)
            if (transformedImage is not None):
                cv2.imwrite(os.path.join(destinationFolder, filename), transformedImage)
            print(filename, " transform finished.", sep="")



    @staticmethod
    def TransformImage(sourceImage : Image, targetImageSize: int = 256, syncedImages : [Image] = None) -> Image:
        '''
        Метод преобразования изображения
        :param sourceImage: Исходное изображение
        :param targetImageSize: Размер изображения (оно будет квадратным)
        :param syncedImages: Изображения, которые нужно преобразовать синхронно с исходным (например, различные маски)
        :return: Преобразованное изображение (или None при некорретных данных)
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
        mask = sourceImageArray <= 20
        blackPixels = np.all(mask, axis = 2)

        blackRows = np.all(blackPixels, axis = 1)
        blackColumns = np.all(blackPixels, axis = 0)

        # Выделяем границы глаза на исходном изображении
        nonBlackIndices = np.nonzero(np.logical_not(blackRows))
        if (np.size(nonBlackIndices) == 0):
            return None

        rowMin = nonBlackIndices[0][0]
        rowMax = nonBlackIndices[0][-1]

        nonBlackIndices = np.nonzero(np.logical_not(blackColumns))
        if (np.size(nonBlackIndices) == 0):
            return None

        columnMin = nonBlackIndices[0][0]
        columnMax = nonBlackIndices[0][-1]

        destinationImageArray = DataTransformer.GetSquaredImage(sourceImageArray, rowMin, rowMax, columnMin, columnMax)
        destinationImageArray = DataTransformer.PreprocessImage(destinationImageArray)

        mainImg = PIL.Image.fromarray(destinationImageArray, mode="RGB").resize((targetImageSize, targetImageSize))
        if syncedImages is None:
            return mainImg

        syncedImgs = []
        for syncImg in syncedImages:
            syncImgArray = np.asarray(syncImg).astype(np.uint8)
            if (syncImg.format == "TIFF"):
                syncImgArray *= 255

            if (syncImgArray.ndim == 2):
                syncImgArray = np.expand_dims(syncImgArray, 2)
            m1 = syncImgArray.min()
            m2 = syncImgArray.max()
            croppedSyncImg = DataTransformer.GetSquaredImage(syncImgArray, rowMin, rowMax, columnMin, columnMax)
            croppedSyncImg = DataTransformer.PreprocessImage(croppedSyncImg)
            syncedImgs.append(PIL.Image.fromarray(croppedSyncImg, mode="RGB").resize((targetImageSize, targetImageSize)))

        return mainImg, syncedImgs

    @staticmethod
    def TransformImage2(sourceImage: Image, targetImageSize: int = 256, syncedImages: [Image] = None) -> np.array:
        '''
        Метод преобразования изображения
        :param sourceImage: Исходное изображение
        :param targetImageSize: Размер изображения (оно будет квадратным)
        :param syncedImages: Изображения, которые нужно преобразовать синхронно с исходным (например, различные маски)
        :return: Преобразованное изображение в виде массива
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
        mask = sourceImageArray <= 20
        blackPixels = np.all(mask, axis=2)

        blackRows = np.all(blackPixels, axis=1)
        blackColumns = np.all(blackPixels, axis=0)

        # Выделяем границы глаза на исходном изображении
        nonBlackIndices = np.nonzero(np.logical_not(blackRows))
        if (np.size(nonBlackIndices) == 0):
            return None

        rowMin = nonBlackIndices[0][0]
        rowMax = nonBlackIndices[0][-1]

        nonBlackIndices = np.nonzero(np.logical_not(blackColumns))
        if (np.size(nonBlackIndices) == 0):
            return None

        columnMin = nonBlackIndices[0][0]
        columnMax = nonBlackIndices[0][-1]

        destinationImageArray = DataTransformer.GetSquaredImage(sourceImageArray, rowMin, rowMax, columnMin, columnMax)
        destinationImageArray = DataTransformer.PreprocessImage(destinationImageArray)

        mainImg = cv2.resize(destinationImageArray, dsize = (targetImageSize, targetImageSize))#PIL.Image.fromarray(destinationImageArray, mode="F").resize((targetImageSize, targetImageSize))
        if syncedImages is None:
            return mainImg

        syncedImgs = []
        for syncImg in syncedImages:
            syncImgArray = np.asarray(syncImg).astype(np.uint8)
            if (syncImg.format == "TIFF"):
                syncImgArray *= 255

            if (syncImgArray.ndim == 2):
                syncImgArray = np.expand_dims(syncImgArray, 2)
            m1 = syncImgArray.min()
            m2 = syncImgArray.max()
            croppedSyncImg = DataTransformer.GetSquaredImage(syncImgArray, rowMin, rowMax, columnMin, columnMax)
            croppedSyncImg = DataTransformer.PreprocessImage(croppedSyncImg)
            syncedImgs.append(
                cv2.resize(croppedSyncImg, dsize=(targetImageSize, targetImageSize)))
                #PIL.Image.fromarray(croppedSyncImg, mode="RGB").resize((targetImageSize, targetImageSize)))

        return mainImg, syncedImgs


    @staticmethod
    def GetSquaredImage(imgArray, rowMin, rowMax, columnMin, columnMax):
        '''
        Метод обрезки изображения по данным граничным индексам с приведением к квадратному виду
        :param imgArray: Массив пикселей
        :param rowMin: Номер первой строки, которая попадёт в изображение
        :param rowMax: Номер последней строки, которая попадёт в изображение
        :param columnMin: Номер первого столбца, который попадёт в изображение
        :param columnMax: Номер последнего столбца, который попадёт в изображение
        :return: Квадратный массив пикселей, обрезанный с учётом переданных параметров
        '''
        rowDelta = rowMax - rowMin
        columnDelta = columnMax - columnMin
        delta = rowDelta - columnDelta

        if delta > 0:
            # Глаз обрезан сверху-снизу, добавляем чёрные строки
            destinationImageArray = np.zeros([rowDelta + 1, rowDelta + 1, 3], dtype=np.uint8)
            destinationImageArray[:, (delta // 2):((delta // 2) + columnDelta + 1), :] = \
                imgArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]
        elif delta < 0:
            # Глаз обрезан слева-справа, добавляем чёрные колонки
            destinationImageArray = np.zeros([columnDelta + 1, columnDelta + 1, 3], dtype=np.uint8)
            destinationImageArray[(-delta // 2):((-delta // 2) + rowDelta + 1), :, :] = \
                imgArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]
        else:  # delta == 0
            # Глаз занял ровно квадратную область, ничего не добавляем
            destinationImageArray = imgArray[rowMin:(rowMax + 1), columnMin:(columnMax + 1), :]

        return destinationImageArray

    @staticmethod
    def PreprocessImage(x):
        x = x / 255.0
        #sharpenedX = 0.5 + 4 * (currentLayer - gaussian_filter(currentLayer, sigma = 10, radius=15))
        sharpenedX = 0.5 + 4 * (x - cv2.GaussianBlur(x, (0, 0), sigmaX=20))

        # Ищем информативные пиксели по строкам и по столбцам (так выделим границы глаза)
        mask = x >= 0.075
        blackPixels = np.all(mask, axis=2)

        blackRows = np.all(blackPixels, axis=1)
        blackColumns = np.all(blackPixels, axis=0)

        # Выделяем границы глаза на исходном изображении
        nonBlackIndices = np.nonzero(np.logical_not(blackRows))
        if (np.size(nonBlackIndices) == 0):
            return None

        rowMin = nonBlackIndices[0][0]
        rowMax = x.shape[0] - 1 - nonBlackIndices[0][-1]

        nonBlackIndices = np.nonzero(np.logical_not(blackColumns))
        if (np.size(nonBlackIndices) == 0):
            return None

        columnMin = nonBlackIndices[0][0]
        columnMax = x.shape[1] - 1 - nonBlackIndices[0][-1]

        indices = np.stack(np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]))).astype(np.float32) - \
                  np.reshape([(x.shape[0] - 1) / 2, (x.shape[0] - 1) / 2], (2, 1, 1))
        indicesNorms = np.linalg.norm(indices, axis=0, ord=2)

        radius = 0.9 * 0.5 * (x.shape[0] - np.maximum(rowMin + rowMax, columnMin + columnMax))
        normalizedRadius = radius / x.shape[0]
        radialMask = np.expand_dims((indicesNorms < radius).astype(np.float32), axis=-1)

        maskedSharpenedX = sharpenedX * radialMask + 0.5 * (1 - radialMask)
        minBox = ((0.5 - normalizedRadius) * np.shape(x)[0]).astype(np.int32)
        maxBox = ((0.5 + normalizedRadius) * np.shape(x)[0]).astype(np.int32)
        croppedMaskedSharpenedX = DataTransformer.GetSquaredImage(maskedSharpenedX, minBox, maxBox, minBox, maxBox)
            #np.squeeze(tf.image.crop_and_resize(np.expand_dims(maskedSharpenedX, axis=0), [
            #[0.5 - normalizedRadius, 0.5 - normalizedRadius, 0.5 + normalizedRadius, 0.5 + normalizedRadius]], [0],
            #                                                          constants.IMAGE_SIZE), axis=0)
        return croppedMaskedSharpenedX * 255.0




    @staticmethod
    def PrepareDataWithSyncedImages(sourceFolder : str, destinationFolders : [str], syncedFolders : [str] = None, targetImageSize : int = 256, nThreads : int = 1):
        '''
        Метод преобразования изображений в формат для нейронок
        :param sourceFolder: Папка с исходными изображениями
        :param destinationFolders: Папки, куда складываются преобразованные изображения (м. б. несколько, см. ниже)
        :param syncedFolders: Папки с изображениями, которые надо преобразовать синхронно с основными
        :param targetImageSize: Размер преобразованного изображения
        :param nThreads: Количество потоков для разбиения
        :return:
        '''
        if syncedFolders is None:
            if len(destinationFolders) != 1:
                print("Destination folder must be single folder if no synced are given")
                return
        else:
            if len(destinationFolders) != len(syncedFolders) + 1:
                print("Destination folders size does not match given synced folders size")
                return

        print("Transform data (with synced) method entered")
        validExtensions = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
        mainFilenames = []
        print("Preparing main filenames")
        for filename in os.listdir(sourceFolder):
            imageExtension = os.path.splitext(filename)[1]
            if imageExtension.lower() not in validExtensions:
                continue

            mainFilenames.append(filename)

        #
        syncedFilenames = []
        nFiles = len(mainFilenames)
        nSyncedFolders = len(syncedFolders)
        print("Preparing synced filenames")

        # Подготавливаем список всех файлов
        folderFilenames = []
        for i in range(nSyncedFolders):
            folderFilenames.append([filename for filename in os.listdir(syncedFolders[i])])


        # Переформируем массив так, чтобы каждый внутренний элемент содержал список изображенный,
        # синхронных с исходным, лежащим по тому же индексу
        for i in range(nFiles):
            names = []
            for j in range(nSyncedFolders):
                names.append(folderFilenames[j][i])

            syncedFilenames.append(names)
            #syncedFilenames.append([filename for filename in folderFilenames[:][i]])

        print("Transforming image tuples")


        for i in range(len(mainFilenames)):
            mainImg = PIL.Image.open(sourceFolder + mainFilenames[i])
            syncedImgs = [PIL.Image.open(syncedFolders[j] + syncedFilenames[i][j]) for j in range(len(syncedFolders))]

            transformedMainImg, transformedSyncImgs = DataTransformer.TransformImage(mainImg, 512, syncedImgs)

            if (transformedMainImg is not None):
                transformedMainImg.save(os.path.join(destinationFolders[0], mainFilenames[i]))

            for j in range(len(syncedFolders)):
                if (transformedSyncImgs[j] is not None):
                    transformedSyncImgs[j].save(os.path.join(destinationFolders[1 + j], syncedFilenames[i][j]))


        print("Transform data (with synced) method exited")

    @staticmethod
    def OrganizeImages(sourceFolder: str, label_csv_filename : str):
        '''
        Создание структуры папок для организованного доступа к датасету
        :param sourceFolder: Расположение исходных файлов
        :return:
        '''

        labelFile = read_csv(label_csv_filename)
        labelFilenames = labelFile.to_numpy()[:, 0]
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

            labelIndex = np.where(labelFilenames == os.path.splitext(imageFilenames[index])[0])[0][0]
            imagePath = sourceFolder + "test/" + classSubfolders[labels[labelIndex]] + imageFilenames[index]
            print(imageFilenames[index], " has grade ", labels[labelIndex], sep="")

            image = PIL.Image.open(sourceFolder + imageFilenames[index])
            image.save(imagePath)
            os.remove(sourceFolder + imageFilenames[index])
            print(imageFilenames[index], " organize finished.", sep = "")

        for index in indices:
            print("Working with ", imageFilenames[index], "...", sep="")

            labelIndex = np.where(labelFilenames == os.path.splitext(imageFilenames[index])[0])[0][0]
            imagePath = sourceFolder + "train/" + classSubfolders[labels[labelIndex]] + imageFilenames[index]
            print(imageFilenames[index], " has grade ", labels[labelIndex], sep="")

            image = PIL.Image.open(sourceFolder + imageFilenames[index])
            image.save(imagePath)
            os.remove(sourceFolder + imageFilenames[index])
            print(imageFilenames[index], " organize finished.", sep = "")


    @staticmethod
    def BalanceImages(sourceFolder: str, labelCsvFilename : str, destinationFolder: str, weights):
        '''
        Удаление части данных до полного совпадения количества образцов в классах
        :param sourceFolder: Исходная папка с данными (где лежат все данные вперемешку)
        :param labelCsvFilename: Файл с метками для каждого образца
        :param destinationFolder: Папка, где будут сложены данные (сгруппированы по папкам) и файл с метками для усечённой выборки
        :param weights: Список желаемых весов (относительное число образцов)
        :return:
        '''
        labelFile = read_csv(labelCsvFilename)
        labelFile = labelFile.reset_index(drop=True)
        counts = labelFile.value_counts("level")

        weights = np.array(weights)
        weights = weights / sum(weights)
        weights = ((weights / min(weights)) * min(counts.to_numpy())).astype(np.int32)

        newLabelFile = pd.DataFrame(columns =["image", "level"])

        counter = 0
        for i in range(5):
            temp1 = labelFile.loc[labelFile["level"] == i]
            filenames = temp1["image"].to_numpy()
            filenames = np.random.choice(filenames, weights[i], replace = False)
            for filename in filenames:
                counter = counter + 1
                dirCounter = len([i for i in os.listdir(destinationFolder)])
                if dirCounter != counter - 1:
                    dummy = True

                if not os.path.isfile(sourceFolder + filename + ".jpeg"):
                    print("INVALID FILE")

                print("#", counter, "/", dirCounter, " Working with ", filename, "...", sep="")
                img = Image.open(sourceFolder + filename + ".jpeg")
                img.save(destinationFolder + filename + ".jpeg")
                dummy = labelFile.loc[labelFile["image"] == filename]
                newLabelFile = pd.concat([newLabelFile, dummy])
                #time.sleep(0.1)

        newLabelFile.to_csv(destinationFolder + "labels.csv", index = False)

