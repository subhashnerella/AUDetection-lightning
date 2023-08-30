from main import DataConfig
from main import instantiate_from_config
import argparse
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
import glob
from data.base import ImagePaths
import torch
from einops import rearrange
import numpy as np
import os
import pandas as pd

def arg_parser():
    parser = argparse.ArgumentParser('CNN: AU detector', add_help=False)

    parser.add_argument(
                        "-b",
                        "--base",
                        nargs="*",
                        metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=list(),
                        )
    return parser


def main():
    parser = arg_parser()
    opt = parser.parse_args()

    configs = [OmegaConf.load(base) for base in opt.base]
    config = OmegaConf.merge(*configs)

    model = instantiate_from_config(config.model)
    model.eval()
    model.cuda()


    data_sources = ['/data/datasets/I2CU_processedFaceVideos/', '/data/datasets/PAIN_processedFaceVideos/']
    aus = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU9', 'AU10','AU12', 'AU14', 'AU15', 'AU17','AU20', 'AU23', 'AU24', 'AU25','AU26','AU27','AU43']
    size = 224
    file_paths = []
    for data_source in data_sources:
        paths = glob.glob(data_source+'*/*/extracted_frames/*.jpg')
        file_paths.extend(paths)

    save_file ='inference/ICU_AUs.csv'
    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
    else:
        df = pd.DataFrame(columns=['patient','video','frame','file_path']+aus)

    if len(df)>0:
        lastrunfile = df['file_path'].values[-1]
        ind = file_paths.index(lastrunfile)
        file_paths = file_paths[ind+1:]
    file_paths = list(set(file_paths) - set(df['file_path'].values))
    #yeilds chunk of 30 images, drop images that fail the detector
    data = ImagePaths(file_paths,landmark_paths=None,aus=aus,size=size)
    
    length = len(file_paths)

    images = []
    paths = []
    try:
        for i in range(0,length):
            if i%10000 == 0:
                print(i)
                df.to_csv(save_file,index=False)
            out = data.__getitem__(i)
            if out['image'] is not None:
                images.append(out["image"])
                paths.append(out["file_path_"])
            if len(images) == 30 or i == length-1:
                images = torch.Tensor(np.stack(images,axis=0))
                images = rearrange(images, 'b h w c -> b c h w')
                #move to gpu
                images = images.to(memory_format=torch.contiguous_format).float()
                images = images.cuda()
                out = model(images).sigmoid()
                out = out.cpu().detach().numpy()
                out = np.around(out,decimals=2)
                patients = [path.split('/')[-4] for path in paths]
                videos = [path.split('/')[-3] for path in paths]
                frames = [path.split('/')[-1].split('.')[0] for path in paths]
                out = np.concatenate([np.array(patients)[:,None],np.array(videos)[:,None],np.array(frames)[:,None],np.array(paths)[:,None],out],axis=1)
                out = pd.DataFrame(out,columns=['patient','video','frame','file_path']+aus)
                df = pd.concat([df,out],axis=0)
                images = []
                paths = []
    except Exception as e:
        
        print(e)
        print('Error occured at index',i)
    finally:
        df.to_csv(save_file,index=False)



    #paths = 
    
    # data = instantiate_from_config(config.data)

    # trainer = Trainer(**config.trainer)
    # trainer.predict(model, data)

if __name__ == '__main__':
    main()