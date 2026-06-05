import os
import json
import argparse
from shutil import copyfile

import torch

import core
import core.trainer
from core.dataset import *
from model.dabit import DaBiT
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher


from torch.utils.tensorboard import SummaryWriter

def main(config : dict):

    if torch.cuda.is_available():
        config["device"] = "cuda"
        print("Using ", torch.cuda.get_device_name(0))
        torch.set_default_device(config["device"])
    else:
        config["device"] = "cpu"
        print("No GPU found, using CPU instead")

    # Dataset and Sampler
    train_args = config["trainer"]
    youtube_vos_train = TrainDataset(config["dl_config"], config["youtube-vos"], config["device"])
    bvidvc_train = TrainDataset(config["dl_config"], config["bvidvc"], config["device"])
    datasets_train = [bvidvc_train, youtube_vos_train]
    train_sampler = Sampler(datasets_train, iter=False)


    # DataLoaders
    dataloader_args = dict(
        dataset=train_sampler,
        batch_size=train_args["batch_size"],
        shuffle=False,
        num_workers=train_args["num_workers"],
        drop_last=True,
    )
    train_loader = PrefetchDataLoader(
        train_args["num_prefetch_queue"], **dataloader_args
    )
    prefetcher = CPUPrefetcher(train_loader)

    # Output directory
    config["out_dir"] = os.path.join(
        config["out_dir"],
        "{}_{}".format(config["net"], os.path.basename(args.config).split(".")[0]),
    )
    os.makedirs(config["out_dir"], exist_ok=True)
    config_path = os.path.join(config["out_dir"], args.config.split("/")[-1])
    if not os.path.isfile(config_path):
        copyfile(args.config, config_path)

    print("==> Created {}".format(config["out_dir"]))


    # Model and Trainer
    model = DaBiT().to(config["device"])
    print(config["net"], "network created")

    trainer = core.trainer.Trainer(config, prefetcher, model)
    trainer.train()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="configs/dabit.json", type=str
    )
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)
