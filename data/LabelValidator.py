import pandas as pd


def ValidateLabelFile(validatingFilename : str, sourceFilename: str):
    validatingDataframe = pd.read_csv(validatingFilename)
    sourceDataframe = pd.read_csv(sourceFilename)

    for rowToValidate in validatingDataframe.iterrows():
        #print(rowToValidate[1]["image"])
        sourceRow = sourceDataframe.loc[sourceDataframe["image"] == rowToValidate[1]["image"]]
        if sourceRow.empty:
            continue

        if sourceRow["level"].iloc[0] != rowToValidate[1]["level"]:
            return False

    return True