from DataTransformer import DataTransformer
from LabelValidator import ValidateLabelFile
import os
import pandas as pd

def main():
    imgs = [i for i in os.listdir("/data/eyepacs/img_trunc/")]
    #labelFile = pd.read_csv("D:/Study/Magister/CourseworkProject/main/data/img_trunc/labels.csv")

    #for segmentation in labelFile.iterrows():
    #    filename = segmentation[1]["image"]
    #    if (imgs.index(filename + ".jpeg") == -1):
    #        print(f"File {filename} not found")

    #DataTransformer.BalanceImages("D:/Study/Magister/CourseworkProject/main/data/img_full/img/",
    #                               "D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainLabels.csv",
    #                              "D:/Study/Magister/CourseworkProject/main/data/img_trunc/",
    #                              [1.2, 1.2, 1.1, 1.1, 1])
    #print(ValidateLabelFile("D:\\Study\\Magister\\CourseworkProject\\main\\data\\trainLabels.csv", "D:\\Study\\Magister\\CourseworkProject\\main\\data\\img_trunc\labels.csv"))
    DataTransformer.OrganizeImages("D:/Study/Magister/CourseworkProject/main/data/img_trunc/",
                                   "/data/eyepacs/img_trunc/labels.csv")


if __name__ == "__main__":
    main()