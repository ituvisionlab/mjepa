# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import torch.version
import yaml

import sys 
sys.path.append('/home/gozde/medChangeDet/jepa')

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from app.vjepa.utils import get_new_log_dir
import torch.utils.tensorboard

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
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        #dump = os.path.join(params['logging']['folder'], 'params-pretrain-mri.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')


    # Tensorboard config
    
    # writer = None
    # if rank == 0:
    #     log_path = "./logs"
    #     log_dir = get_new_log_dir(log_path, prefix=f'{params["app"]}_pretrain_', postfix='')
    #     params['logging']['log_path'] = log_dir
    #     writer = torch.utils.tensorboard.SummaryWriter(params['logging']['log_path'])
    
    
    
        
    # Launch the app with loaded config
    # app_main(params['app'], args=params, log_writer=writer)
    app_main(params['app'], args=params, log_dir=log_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    gpu_devices = args.devices
    
    num_gpus = len(args.devices)
    
     # Run only one process for debugging
    # process_main(0, args.fname, num_gpus, gpu_devices)

    # Tensorboard config
    
    log_dir = get_new_log_dir(args.log_dir, prefix=f'mjepa_pretrain_', postfix='')
    
    mp.set_start_method('spawn')
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_main, args=(rank, args.fname, num_gpus, gpu_devices, log_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
