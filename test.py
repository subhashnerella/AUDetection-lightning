import datetime
import argparse
import os

from main import instantiate_from_config, DataConfig

import lightning.pytorch as pl
from lightning import seed_everything
from lightning.pytorch import Trainer

def arg_parser():
    parser = argparse.ArgumentParser('SWIN: AU detector test',add_help=False)
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "-b",
        "-base",
        type=str,
        default="configs/test.yaml",
    )
    parser.add_argument()
    return parser

def main():
    parser = arg_parser()
    opt = parser.parse_args()
    seed_everything(42)

    if not os.path.exists(opt.checkpoint):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.checkpoint.split("/")
        logdir = "/".join(paths[:-2])
        ckpt = opt.checkpoint
    else:
        assert os.path.isdir(opt.checkpoint), opt.checkpoint
        logdir = opt.checkpoint.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    ckptdir = os.path.join(logdir, "checkpoints")

  