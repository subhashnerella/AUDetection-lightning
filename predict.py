from main import DataConfig
from main import instantiate_from_config
import argparse
from omegaconf import OmegaConf
from lightning.pytorch import Trainer

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
    #paths = 
    
    # data = instantiate_from_config(config.data)

    # trainer = Trainer(**config.trainer)
    # trainer.predict(model, data)

if __name__ == '__main__':
    main()