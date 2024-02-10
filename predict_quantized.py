import datetime
import argparse
import os
import glob
from omegaconf import OmegaConf
from main import instantiate_from_config
from dotenv import load_dotenv
import lightning.pytorch as pl
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from model.swin import SwinModel

import neptune
from neptune import Run 
from callbacks import CUDACallback,PredictLogger

from data.dataset import ICUPred

from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def arg_parser():

    parser = argparse.ArgumentParser('SWIN: AU detector test',add_help=False)
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
                    "-s",
                    "--seed",
                    type=int,
                    default=23,
                    help="seed for seed_everything",
                    )  
    parser.add_argument(
                    "-p",
                    "--data_path",
                    type=str,
                    nargs='+',
                    required=True,
                    )
    return parser


def main():
    parser = arg_parser()
    opt = parser.parse_args()
    seed_everything(opt.seed)

    if not os.path.exists(opt.checkpoint):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.checkpoint):
        paths = opt.checkpoint.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = opt.checkpoint
    else:
        assert os.path.isdir(opt.checkpoint), opt.checkpoint
        logdir = opt.checkpoint.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "bestsofar.ckpt")
    nowname = logdir.split("/")[-1]

    config_files = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    configs = [OmegaConf.load(f) for f in config_files]
    config = OmegaConf.merge(*configs)


    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt)
    #quantize

    backend = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    quantized_model  = torch.quantization.quantize_dynamic(model, qconfig_spec={nn.Linear}, dtype=torch.qint8)
    #scripted_quantized_model = torch.jit.script(quantized_model)
   


    dataset = ICUPred(imgspaths=opt.data_path,size=224)
    dataloader = DataLoader(dataset,batch_size=200,shuffle=False,num_workers=4)


    trainer_config = {'accelerator':'cpu',
                      'devices':1,}
    
    id = config.lightning.logger_id
    run = Run(project='AUdetection',name=nowname,with_id=id,api_token=os.getenv('NEPTUNE_API_KEY'))
    logger = NeptuneLogger(run=run,prefix="testing")

    locallogger = PredictLogger(logdir=logdir)

    trainer = Trainer(logger=logger,callbacks=[locallogger],**trainer_config)
    trainer.predict(model = quantized_model, dataloaders = dataloader)



if __name__ == '__main__':
    load_dotenv()
    main()