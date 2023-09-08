import os
from torch.utils.data import Dataset
import pandas as pd
from data.base import ImagePaths, ConcatDatasetWithIndex
import numpy as np

#ROOT = "/blue/parisa.rashidi/subhashnerella/Datasets/"
ROOT = '/data/datasets/users/subhash/'

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = None

    def _load(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

# ICUOLD##############################################################################################
class ICUTrainOLD(FacesBase):
    def __init__(self,aus,size=225, mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/icu_old.csv'))
        df = helper_icu_split_func(df)
        paths = df['path'].values
        landmark_paths = df['landmark_path'].values
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels,'dataset':'ICUOLD' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

class ICUValOLD(FacesBase):
    def __init__(self,aus,size=225, mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/icu_old.csv'))
        df['path'] = df['path']
        df = helper_icu_split_func(df,split='val')
        paths = df['path'].values
        landmark_paths = df['landmark_path'].values
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels,'dataset':'ICUOLD' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

# ICU##############################################################################################
class ICU(FacesBase):
    def __init__(self,aus,split=None,size=225, mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/icu.csv'))
        df['path'] = df['path'].str.replace('Sampled_Images','extracted_frames')
        if split is not None:
            df = helper_icu_split_func(df,split=split)
        paths = df['path'].values
        landmark_paths = None
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels,'dataset':'ICU' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

# BP4D##############################################################################################
class BP4D(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/bp4d.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'BP4D' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

# DISFA##############################################################################################
class DISFA(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/disfa.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy() 
        labels={'aus':au_labels, 'dataset':'DISFA' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)

# UNBC##############################################################################################
class UNBC(FacesBase):
    def __init__(self,aus,split=None,size=225,mcManager=None):
        super().__init__()
        df = pd.read_csv(os.path.join('data/datafiles/unbc.csv'))
        if split is not None:
            df = helper_split_func(df,split=split)
        relpaths = df['path'].values    
        landmark_paths = df['landmark_path'].values
        paths = list(map(lambda x: os.path.join(ROOT,x),relpaths))
        landmark_paths = list(map(lambda x: os.path.join(ROOT,x),landmark_paths))
        aus_df = helper_AU_func(df,aus)
        au_labels = aus_df[aus].to_numpy()
        labels={'aus':au_labels, 'dataset':'UNBC' }
        self.data = ImagePaths(paths,landmark_paths,aus,labels,size,mcManager)


# MultiDataset##############################################################################################

class MultiDataset(Dataset):
    def __init__(self, datasets,aus,split=None,size=225,mcManager=None):
        dataset_classes = {'BP4D': BP4D,
                           'DISFA': DISFA,
                           'UNBC': UNBC,
                           'ICU': ICU,}
        dataset = []
        for d in datasets:
            dataset.append(dataset_classes[d](aus,split,size=size,mcManager=mcManager))
        self.dataset = ConcatDatasetWithIndex(dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample,_ = self.dataset[idx]
        #sample['dataset_label'] = dataset_label
        return sample
    

# Helper Functions##############################################################################################

def helper_AU_func(df:pd.DataFrame,aus:list)->pd.DataFrame:
    au_df = df.filter(regex='AU*',axis=1)
    present_aus = au_df.columns.to_list()
    to_remove = list(set(present_aus) - set(aus))
    au_df = au_df.drop(columns=to_remove)
    absent_aus = list(set(aus) - set(present_aus))
    # Add absent AUs fillled with -1
    for au in absent_aus:
        au_df[au] = -1
    au_df = au_df[aus]
    return au_df
        
def helper_split_func(df:pd.DataFrame,split:str = 'train')->pd.DataFrame:
    np.random.seed(42)
    participants = df['participant'].unique()
    participants = np.random.choice(participants,size=int(len(participants)*0.75),replace=False)
    
    if split == 'train':
        df = df[df['participant'].isin(participants)]
        print(np.sort(df.participant.unique().tolist()))
    else:
        df = df[~df['participant'].isin(participants)]
        df = df.sample(frac=1)
        print(np.sort(df.participant.unique().tolist()))
    return df

def helper_icu_split_func(df:pd.DataFrame,split:str = 'train')->pd.DataFrame:
    np.random.seed(42)
    patients = df.patient.unique()
    n_patients = len(patients)
    n_images = len(df)
    counts = df.groupby('patient')['patient'].count()
    counts = dict(zip(counts.index,counts.values))
    while True:

        train_patients = np.random.randint(int(0.6*n_patients),int(0.8*n_patients))
        train= np.random.choice(patients,train_patients,replace=False)
        trainimgs = np.array(list(map(lambda x:counts[x],train))).sum()
        if trainimgs > int(0.6*n_images) and trainimgs < int(0.8*n_images):
            break
    if split == 'train':
        df = df[df['patient'].isin(train)]
        print(np.sort(df.patient.unique().tolist()))
    else: 
        df = df[~df['patient'].isin(train)]
        df = df.sample(frac=1)
        print(np.sort(df.patient.unique().tolist()))
    
    return df