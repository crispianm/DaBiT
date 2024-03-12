import os
import json
import argparse
import subprocess

from shutil import copyfile
import torch.distributed as dist

import torch
import torch.multiprocessing as mp

import core
import core.trainer
import core.trainer_flow_w_edge
import core.trainer_depth


from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)


def main(config):

    config["save_dir"] = os.path.join(
        config["save_dir"],
        "{}_{}".format(
            config["model"]["net"], os.path.basename(args.config).split(".")[0]
        ),
    )

    config["save_metric_dir"] = os.path.join(
        "./scores",
        "{}_{}".format(
            config["model"]["net"], os.path.basename(args.config).split(".")[0]
        ),
    )

    if torch.cuda.is_available():
        config["device"] = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        config["device"] = "cpu"
        print("No GPU found, using CPU instead")

    os.makedirs(config["save_dir"], exist_ok=True)
    config_path = os.path.join(config["save_dir"], args.config.split("/")[-1])
    if not os.path.isfile(config_path):
        copyfile(args.config, config_path)
    print("[**] create folder {}".format(config["save_dir"]))

    trainer_version = config["trainer"]["version"]
    trainer = core.__dict__[trainer_version].__dict__["Trainer"](config)

    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="configs/train_depthpainter.json", type=str
    )

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config)
