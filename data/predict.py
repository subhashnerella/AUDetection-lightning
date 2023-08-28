import os
from torch.utils.data import Dataset
import pandas as pd
from data.base import ImagePaths, ConcatDatasetWithIndex
import numpy as np
import glob
from PIL import Image




class ICUPredict(Dataset):
    def __init__(self, data_sources, size=224, mcManager=None):
        self.size = size
        self.file_paths = []
        for data_source in data_sources:
            paths = glob.glob(data_source+'*/*/extracted_frames/*.jpg')
            self.file_paths.extend(paths)
        self._length = len(self.file_paths)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return {'image':file_path}


