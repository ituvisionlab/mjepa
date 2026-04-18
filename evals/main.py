# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import argparse

import multiprocessing as mp
# import debugpy

import pprint
import yaml

import sys 
sys.path.append('/ari/users/eergun01/jepa')

from app.vjepa.utils import get_new_log_dir
from src.utils.distributed import init_distributed

from evals.scaffold import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--log_dir', type=str, default="./logs",
    help='directory path for tensorboard logging'
)
parser.add_argument(
    '--keep_logs',  type=bool, default=False,
    help="Turn logging off by setting it to False"
)

def process_main(rank, fname, world_size, devices, log_dir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    # Your original processing code here
    print(f"Process {rank} running on GPU {devices[rank]}")

    import logging
    #logging.basicConfig()
    # logging.basicConfig(filename='my_log_file.log')
    
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    print("In process_main, before Load Config:")
    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    # Log config
    if rank == 0 and log_dir != None:
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    # NOTE: Esra: Commenting out for debug.

    #world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    world_size, rank = 1, 0
    logger.info(f'Running... (rank: {rank}/{world_size})')

    print("In main.py/process_main, listing params.keys:")
    print(params.keys())
    
    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params, log_dir=log_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    gpu_devices = args.devices

    # Load config (needed for log_dir logic)
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    if args.keep_logs or "tsne" in params["eval_name"].lower():
        log_dir = get_new_log_dir(
            params['logging']['folder'],
            prefix=f"{params['write_tag']}_eval_",
            postfix=''
        )
    else:
        log_dir = None

    # ✅ DEBUG MODE: run single process only
    process_main(
        rank=0,
        fname=args.fname,
        world_size=1,
        devices=[gpu_devices[0]],
        log_dir=log_dir
    )

    sys.exit(0)
