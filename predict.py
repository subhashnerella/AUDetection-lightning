import datetime
import argparse
import os
import glob
from omegaconf import OmegaConf
from main import instantiate_from_config

import lightning.pytorch as pl
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger

import neptune
from neptune import Run 
from callbacks import CUDACallback,PredictLogger

from data.dataset import ICUPred

from torch.utils.data import DataLoader


def arg_parser():
    parser = argparse.ArgumentParser('CNN: AU detector', add_help=False)

    parser = argparse.ArgumentParser('SWIN: AU detector test',add_help=False)
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
    )
    return parser


def main():
    parser = arg_parser()
    opt = parser.parse_args()
    seed_everything(42)

    if not os.path.exists(opt.checkpoint):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.checkpoint):
        paths = opt.checkpoint.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = opt.checkpoint
    else:
        assert os.path.isdir(opt.checkpoint), opt.checkpoint
        logdir = opt.checkpoint.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "epoch=000000.ckpt")
    nowname = logdir.split("/")[-1]

    config_files = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    configs = [OmegaConf.load(f) for f in config_files]
    config = OmegaConf.merge(*configs)


    model = instantiate_from_config(config.model)

    dataset = ICUPred(imgspath='',size=224)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4)


    trainer_config = {'accelerator':'gpu',
                      'devices':1,}
    
    id = config.lightning.logger_id
    run = Run(project='AUdetection',name=nowname,with_id=id,api_token=os.getenv('NEPTUNE_API_KEY'))
    logger = NeptuneLogger(run=run,prefix="testing")

    locallogger = PredictLogger(logdir=logdir)

    trainer = Trainer(logger=logger,callbacks=[locallogger],**trainer_config)
    trainer.predict(model = model, dataloaders = dataloader, ckpt_path = ckpt)



if __name__ == '__main__':
    main()