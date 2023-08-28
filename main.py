import datetime
import argparse
import importlib
from typing import Any, Optional
from omegaconf import OmegaConf
import sys
import os
from dotenv import load_dotenv

from torch.utils.data import DataLoader,DistributedSampler

import lightning
import lightning.pytorch as pl
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.core import LightningDataModule

from callbacks import MetricLogger, SetupCallback, ModelCheckpoint, CUDACallback

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
    parser.add_argument(
                        "-l",
                        "--logdir",
                        type=str,
                        default="logs",
                        help="directory for logging dat shit",
                        )
    parser.add_argument(
                        "-r",
                        "--resume",
                        type=str,
                        const=True,
                        default="",
                        nargs="?",
                        help="resume from logdir or checkpoint in logdir",
                        )
    parser.add_argument(
                        "-s",
                        "--seed",
                        type=int,
                        default=23,
                        help="seed for seed_everything",
                        )  
    return parser

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class DataConfig(LightningDataModule):
    def __init__(self,batch_size,train=None, validation=None, test=None,
                 num_workers=None ):
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.validation = validation
        self.test = test
        self.num_workers = num_workers if num_workers is not None else 8
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self.get_train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self.get_validation_dataloader
        if test is not None:
            self.dataset_configs["test"] = test        
            self.test_dataloader = self.get_test_dataloader                                                                                                              


    def prepare_data(self):
        pass
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def get_train_dataloader(self):
        train_loader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        return train_loader
    
    def get_validation_dataloader(self):
        validation_loader = DataLoader(self.datasets["validation"], batch_size=self.batch_size,
                                       shuffle=False, num_workers=self.num_workers)
        return validation_loader
    
    def get_test_dataloader(self):
        test_loader = DataLoader(self.datasets["test"], batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers)
        return test_loader


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = arg_parser()
    opt = parser.parse_args()
    if opt.resume:
        pass
    else:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = cfg_fname.split('.')[0]
        name = "_"+cfg_name
        nowname = now + name 
        logdir = os.path.join(opt.logdir, nowname)   
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)    

    try:
        configs = [OmegaConf.load(f) for f in opt.base]
        config = OmegaConf.merge(*configs)
        lightning_config = config.lightning
        trainer_config = lightning_config.trainer

        model = instantiate_from_config(config.model)
        data = instantiate_from_config(config.data)
        #data.setup()

        trainer_kwargs = dict()


        #############################################################################################

        default_modelckpt_cfg = {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "save_top_k": 1,
                "monitor": "val/loss",
            }
        }

        #############################################################################################

        default_callbacks_cfg = {
            "setup_callback": {
                "target": "callbacks.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
             },  
            "metric_logger": {
                "target": "callbacks.MetricLogger",
                "params": {
                    "logdir": logdir,
                }

            },
            "cuda_callback": {
                'target': "callbacks.CUDACallback",
            }
        }
        default_callbacks_cfg.update({'checkpoint_callback': default_modelckpt_cfg})
        callbacks_cfg = OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        #############################################################################################
                
        
        default_logger_cfgs = {
                        "tensorboard": {
                            "target": "lightning.pytorch.loggers.TensorBoardLogger",
                            "params": {
                                "name": nowname,
                                "save_dir": opt.logdir,
                            }   
                                     },
                        "wandb": {
                                "target": "lightning.pytorch.loggers.WandbLogger",
                                "params": {
                                        "name": nowname,
                                        "save_dir": logdir,
                                        }
                                 },
                        "neptune": {
                                "target": "lightning.pytorch.loggers.NeptuneLogger",
                                "params": {
                                        "api_key": os.getenv('NEPTUNE_API_KEY'),
                                        "project": 'AUdetection',
                        }}
                        
        }
        default_logger_cfg = default_logger_cfgs["neptune"]
        logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        #############################################################################################
       
        trainer = Trainer(**trainer_config, **trainer_kwargs)
        trainer.fit(model, data)

        trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path)

    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    load_dotenv()
    main()
        


