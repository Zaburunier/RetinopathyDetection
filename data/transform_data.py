from DataTransformer import DataTransformer
from transform_labels import TransformCSVLabels

def main():
    #DataTransformer.PrepareData(
    #    "D:\Study\Magister\CourseworkProject\Структуризация\Датасеты\Retinopathy (Kaggle)\Retinopathy (Kaggle)",
    #    "D:\Study\Magister\CourseworkProject\main\data\eyepacs\img_preprocessed",
    #    256, 12)

    #DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/eyepacs/img_preprocessed/",
    #                                "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainLabels.csv")

    #DataTransformer.BalanceImages("D:/Study/Magister/CourseworkProject/main/data/eyepacs/img_preprocessed/unorganized/",
    #                                "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainLabels.csv",
    #                              "D:/Study/Magister/CourseworkProject/main/data/eyepacs/img_preprocessed_trunc/",
    #                              [1, 1, 1, 1, 1])

    DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/eyepacs/img_preprocessed_trunc/",
                                    "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainLabels.csv")

    '''
    DataTransformer.PrepareDataOnMultipleProcesses("D:\Study\Magister\CourseworkProject\Структуризация\Датасеты\Retinopathy (Kaggle)\Retinopathy (Kaggle)",
                                "D:\Study\Magister\CourseworkProject\main\data\img2",
                                256, 12)
    DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/img_trunc/",
                                   "D:\\Study\\Magister\\CourseworkProject\\main\\data\\img_trunc\\labels.csv")

    DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/img2/", "D:\\Study\\Magister\\CourseworkProject\\main\\data\\new_trainLabels.csv")

    TransformCSVLabels(["D:/Study/Magister/CourseworkProject/main/data/img2/train/0. No DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/train/1. Mild DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/train/2. Moderate DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/train/3. Severe DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/train/4. Profilerative DR/"],
                        "D:/Study/Magister/CourseworkProject/main/data/trainLabels.csv",
                        "D:/Study/Magister/CourseworkProject/main/data/trainImg2Labels.csv")

    TransformCSVLabels(["D:/Study/Magister/CourseworkProject/main/data/img2/test/0. No DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/test/1. Mild DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/test/2. Moderate DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/test/3. Severe DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img2/test/4. Profilerative DR/"],
                        "D:/Study/Magister/CourseworkProject/main/data/trainLabels.csv",
                        "D:/Study/Magister/CourseworkProject/main/data/testImg2Labels.csv")
    '''
    '''DataTransformer.PrepareDataWithSyncedImages("D:\\Torrent\\idrid\\idrid\\A. Segmentation\\1. Original Images\\b. Testing Set\\",
                                                ["D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\img\\",
                                                 "D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\mask\\ma\\",
                                                 "D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\mask\\ha\\",
                                                 "D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\mask\\he\\",
                                                 "D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\mask\\se\\",
                                                 "D:\\Study\\Magister\\CourseworkProject\\main\\data\\idrid\\test\\mask\\od\\"],
                                                ["D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\1. Microaneurysms\\",
                                                 "D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\2. Haemorrhages\\",
                                                 "D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\3. Hard Exudates\\",
                                                 "D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\4. Soft Exudates\\",
                                                 "D:\\Torrent\\idrid\\idrid\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\5. Optic Disc\\"])'''

if __name__ == "__main__":
    main()