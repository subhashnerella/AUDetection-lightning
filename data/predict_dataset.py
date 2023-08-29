import os
from torch.utils.data import Dataset
import pandas as pd
from data.base import ImagePaths
import numpy as np
import glob
from PIL import Image
from data.dataset import FacesBase



class ICUPredict(FacesBase):
    def __init__(self, data_sources,aus, size=224,):
        self.size = size
        file_paths = []
        for data_source in data_sources:
            paths = glob.glob(data_source+'*/*/extracted_frames/*.jpg')
            file_paths.extend(paths)
        self.data = ImagePaths(file_paths,landmark_paths=None,aus=aus,size=size)


