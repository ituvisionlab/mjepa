# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml
import os

import sys 
sys.path.append('/home/gozde/medChangeDet/jepa')
sys.path.append('/gpfs/home/unalg01/jepa')

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from app.vjepa.utils import get_new_log_dir

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
    '--keep_logs',  type=bool, default=True,
    help="Turn logging off by setting it to False"
)


def process_main(rank, fname, world_size, devices, log_dir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # Log config
    if rank == 0 and log_dir != None:
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    
    # Launch the app with loaded config
    app_main(params['app'], args=params, log_dir=log_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    gpu_devices = args.devices
    
    num_gpus = len(args.devices)
    
     # Run only one process for debugging
    # process_main(0, args.fname, num_gpus, gpu_devices)

    # Logging config
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    
    if args.keep_logs:
        log_dir = get_new_log_dir(params["logging"]['folder'], prefix=f'mjepa_pretrain_', postfix='')
        
        # Wand initialization
        # run = wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="mjepa-project",
            
        #     entity="mgulsen2020-wandb",
            
        #     dir=os.path.join(log_dir),

        #     # track hyperparameters and run metadata
        #     config=params,
            
        #     name=os.path.basename(log_dir),
            
        #     group="mjepa-DDP")    
        
    else:
        log_dir = None
        # run = None
    
    mp.set_start_method('spawn')
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_main, args=(rank, args.fname, num_gpus, gpu_devices, log_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
