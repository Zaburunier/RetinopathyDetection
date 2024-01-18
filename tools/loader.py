import torch
from torch.utils.data import Dataset
import torchvision.transforms as t
from PIL import Image
import pandas as pd
import numpy as np
from loguru import logger as printer


class ODIR5K(Dataset):

    def __init__(self, data_path: str, annotation_path: str, train_test_size: float, is_train: bool,
                 augment: bool = False):
        self.labels = None
        self.data_path = data_path
        self.annotation_path = annotation_path

        def set_filepath(x):
            return data_path + "/" + x

        def set_label(x):
            arr = ','.join(e for e in x if e.isalnum()).split(",")
            arr = np.asarray(arr).astype(int)
            result = np.where(arr == 1)
            return result[0][0]

        df = pd.read_csv(annotation_path)[["target", "filename"]]
        df = df.sample(frac=1)
        df.filename = df.filename.apply(set_filepath)
        df.target = df.target.apply(set_label)

        if is_train:
            set_size = int(df.target.shape[0] * train_test_size)
            df = df.head(set_size)
            printer.info(f"Размер сети: {set_size}")
            printer.info(f"Обучение сети: {df.target.value_counts()}")

        else:
            set_size = int(df.target.shape[0] * (1 - train_test_size))
            df = df.tail(set_size)
            printer.info(f"Проверит размера сети: {set_size}")
            printer.info(f"Провереить распределение сети: {df.target.value_counts()}")

        self.df = df

        if augment is None:
            self.img_transform = t.Compose([
                t.Resize((224, 224)),
                t.ToTensor(),
                t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif augment == "Imagenet":
            self.img_transform = t.Compose([
                t.AutoAugment(t.AutoAugmentPolicy.IMAGENET),
                t.ToTensor(),
                t.Resize((224, 224))
            ])
        elif augment == "Cigar10":
            self.img_transform = t.Compose([
                t.AutoAugment(t.AutoAugmentPolicy.CIFAR10),
                t.ToTensor(),
                t.Resize((224, 224))
            ])

        else:
            raise Exception("Аугментация не поддерживается")

    def __len__(self):
        return self.df.target.shape[0]

    def __getitem__(self, identifier):
        target, filename = self.df.iloc[identifier]
        img = self.img_transform(Image.open(filename))
        label = torch.tensor(target)
        out = {"data": img, "label": label}
        return out

    def make_balanced(self):
        balanced_no_beard = self.labels[self.labels["No_Beard"] == 0].sample(n=2000, random_state=1)
        self.labels = self.labels.drop(self.labels[self.labels.No_Beard == 0].index)
        self.labels = pd.concat([balanced_no_beard, self.labels])
