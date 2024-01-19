import os
import numpy as np
import PIL.Image


def FillAbsentImages(folder : str, namePrefix : str, nameSuffix : str, nImages, startIndex = 1):
    '''
    Заполнение папок пустыми изображениями там, где данных нет
    :param folder: Папка с изображениями
    :param namePrefix: Общий префикс имени изображений (к нему приставляется числовой индекс для проверки наличия)
    :param nameSuffix: Общий постфикс имени изображения (приставляется к имени сохраняемых изображений после числового индекса)
    :param nImages: Требуемое число изображений
    :return:
    '''
    if nImages < 1:
        return

    filenames = [i for i in os.listdir(folder)]

    if len(filenames) == nImages:
        print("No absent images found")
        return

    imageExtension = os.path.splitext(filenames[0])[1]

    sourceImage = PIL.Image.open(folder + filenames[0])
    sourceArray = np.asarray(sourceImage).astype(np.uint8)
    sourceArray[:, :] = 0
    fillerImage = PIL.Image.fromarray(sourceArray)


    for i in range(startIndex, startIndex + nImages):
        soughtPrefix = namePrefix + ("" if i >= 10 else "0") + str(i)
        for name in filenames:
            if name.find(soughtPrefix) >= 0:
                # Нашли изображения, добавлять нечего
                break
        else:
            print("Creating filler image", folder + soughtPrefix + nameSuffix + imageExtension)
            fillerImage.save(folder + soughtPrefix + nameSuffix + imageExtension)



def main():
    FillAbsentImages("D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\4. Soft Exudates\\", "IDRiD_", "_SE", 27, 55)


if __name__ == "__main__":
    main()

