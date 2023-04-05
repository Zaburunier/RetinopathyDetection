import os
import pandas as pd


def TransformCSVLabels(sourceFolders : [str], filepath: str, newFilePath: str):
    labelFile = pd.read_csv(filepath)
    transformedLabelFile = pd.DataFrame(columns = ["image", "level"])

    for folder in sourceFolders:
        for filename in os.listdir(folder):
            filename = os.path.splitext(filename)[0]
            dummy = labelFile.loc[labelFile["image"] == filename]
            #transformedLabelFile = transformedLabelFile.append(dummy)
            transformedLabelFile = pd.concat([transformedLabelFile, dummy])

    transformedLabelFile.to_csv(newFilePath, index = False)


def main():
    TransformCSVLabels(["img"], "trainLabels.csv", "new_TrainLabels.csv")


if __name__ == "__main__":
    main()