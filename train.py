
import os
import json
import argparse
from datetime import timedelta
from shutil import copyfile

import torch
import torch.distributed as dist

import core.trainer
from core.dataset import TrainDataset, Sampler
from model.dabit import DaBiT
from model.misc import set_random_seed
from torch.utils.data import DataLoader
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher


def main(config : dict):

    # Multi-GPU: launched via torchrun, one process per GPU. The generous
    # timeout covers rank 0 running validation while the others wait on the
    # next gradient allreduce.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_main = rank == 0

    # Seed python/numpy/torch for reproducible sampling and init. DataLoader
    # workers derive their own per-worker seeds from torch's base seed. The
    # rank offset makes each DDP process sample different videos.
    set_random_seed(config.get("seed", 2023) + rank)

    if torch.cuda.is_available():
        config["device"] = "cuda"
        if is_main:
            print("Using ", torch.cuda.get_device_name(0), f"x{world_size}")
        torch.set_default_device(config["device"])
    else:
        config["device"] = "cpu"
        print("No GPU found, using CPU instead")

    # Dataset and Sampler
    train_args = config["trainer"]
    dataset_keys = config.get("datasets", ["bvidvc", "youtube-vos"])
    datasets_train = [
        TrainDataset(config["dl_config"], config[key]) for key in dataset_keys
    ]
    print("Training on datasets:", ", ".join(dataset_keys))
    train_sampler = Sampler(datasets_train, iter=False)


    # DataLoaders. Workers build samples on CPU (focal-blur synthesis included)
    # in parallel; the trainer moves each batch to the GPU. The main process sets
    # the default device to cuda, which forked workers inherit, so worker_init_fn
    # forces the CPU default inside each worker (otherwise torch.tensor(...) would
    # try to allocate on the GPU and crash the worker).
    num_workers = train_args["num_workers"]

    def worker_init_fn(worker_id):
        torch.set_default_device("cpu")

    dataloader_args = dict(
        dataset=train_sampler,
        batch_size=train_args["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    if num_workers > 0:
        # PyTorch's own multi-process prefetching (prefetch_factor) feeds the GPU;
        # the legacy threaded PrefetchDataLoader is incompatible with worker
        # multiprocessing (corrupts DataLoader _task_info), so use a plain loader.
        dataloader_args.update(
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=train_args.get("prefetch_factor", 2),
        )
        train_loader = DataLoader(**dataloader_args)
    else:
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
    if is_main:
        config_path = os.path.join(config["out_dir"], args.config.split("/")[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print("==> Created {}".format(config["out_dir"]))


    # Model and Trainer
    model = DaBiT().to(config["device"])
    if is_main:
        print(config["net"], "network created")

    trainer = core.trainer.Trainer(config, prefetcher, model)
    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="configs/dabit.json", type=str
    )
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)
