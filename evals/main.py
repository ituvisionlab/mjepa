# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp
# import debugpy

import pprint
import yaml

import sys 
sys.path.append('/gpfs/home/unalg01/jepa')
sys.path.append('/home/gozde/medChangeDet/jepa')

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

def process_main(rank, fname, world_size, devices, log_dir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    # Your original processing code here
    print(f"Process {rank} running on GPU {devices[rank]}")

    import logging
    #logging.basicConfig()
    logging.basicConfig(filename='my_log_file.log')
    
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
    if rank == 0:
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    print("In main.py/process_main, listing params.keys:")
    print(params.keys())
    
    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params, log_dir=log_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    # args.devices will now be a list of devices directly
    gpu_devices = args.devices
    #num_gpus = len(args.devices)
    num_gpus = len(gpu_devices)
    
    # Run only one process for debugging
    # # process_main(0, args.fname, num_gpus, gpu_devices)
    
    # Load config
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    log_dir = get_new_log_dir(params['logging']['folder'], prefix=f'mjepa_eval_', postfix='')

    mp.set_start_method('spawn')
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_main, args=(rank, args.fname, num_gpus, gpu_devices, log_dir))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        