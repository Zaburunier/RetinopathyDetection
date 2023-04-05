from DataTransformer import DataTransformer
from transform_labels import TransformCSVLabels

def main():
    #DataTransformer.PrepareData("D:\Study\Магистратура\Проект\Структуризация\Датасеты\Retinopathy (Kaggle)\Retinopathy (Kaggle)",
    #                            "D:\Study\Магистратура\Проект\main\data\img",
    #                            256)
    DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/img/", "D:\\Study\\Magister\\CourseworkProject\\main\\data\\new_trainLabels.csv")
    TransformCSVLabels(["D:/Study/Magister/CourseworkProject/main/data/img/train/0. No DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/train/1. Mild DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/train/2. Moderate DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/train/3. Severe DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/train/4. Profilerative DR/"],
                        "D:/Study/Magister/CourseworkProject/main/data/trainLabels.csv",
                        "D:/Study/Magister/CourseworkProject/main/data/trainImgLabels.csv")

    TransformCSVLabels(["D:/Study/Magister/CourseworkProject/main/data/img/test/0. No DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/test/1. Mild DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/test/2. Moderate DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/test/3. Severe DR/",
                        "D:/Study/Magister/CourseworkProject/main/data/img/test/4. Profilerative DR/"],
                        "D:/Study/Magister/CourseworkProject/main/data/trainLabels.csv",
                        "D:/Study/Magister/CourseworkProject/main/data/testImgLabels.csv")


if __name__ == "__main__":
    main()