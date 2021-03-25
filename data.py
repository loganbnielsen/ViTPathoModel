import log
import logging
logger = logging.getLogger('root') 

import torch
from torch import Tensor

from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from os import path
from PIL import Image


class ThoracicDataset(Dataset):
    def __init__(self, summary_csv, root_dir, transform=None):
        self.summary_df = pd.read_csv(path.join(root_dir, summary_csv))
        self.root_dir = root_dir
        self.transform = transform
        self.summary_csv = summary_csv

    def read_img(self, path):
        img = Image.open(path)
        img.load()
        return np.asarray(img, dtype="int32")

    def get_med_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = path.join(self.root_dir, self.summary_df.iloc[idx,0])
        img = self.read_img(img_path)

        if self.transform: 
            img = self.transform(img)

        class_name = self.summary_df.iloc[idx,1]
        class_id   = self.summary_df.iloc[idx,2]
        rad_id     = self.summary_df.iloc[idx,3]
        x_min = self.summary_df.iloc[idx,4]
        y_min = self.summary_df.iloc[idx,5]
        x_max = self.summary_df.iloc[idx,6]
        y_max = self.summary_df.iloc[idx,7]

        return Tensor(img).permute(2,0,1), Tensor([class_id]), Tensor([x_min, y_min, x_max, y_max]), class_name, rad_id

    def __len__(self):
        return len(self.summary_df)


    def __getitem__(self, idx):
        X, class_id, bbox, class_name, rad_id = self.get_med_image(idx)
        return X, class_id, bbox, class_name, rad_id




