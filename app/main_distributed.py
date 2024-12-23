# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import pprint
import yaml

import submitit

import logging
import sys
import traceback
from collections import OrderedDict

import sys 
sys.path.append('/gpfs/home/unalg01/jepa')

from app.scaffold import main as app_main
from src.utils.logging import get_logger
from app.vjepa.utils import get_new_log_dir

#logger = get_logger(force=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
    handlers=[
        logging.FileHandler("job_submission.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OrderedLoader(yaml.SafeLoader):
    pass

def construct_ordered_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))

OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_ordered_mapping
)

class OrderedDumper(yaml.SafeDumper):
    pass

def represent_ordered_mapping(dumper, data):
    return dumper.represent_dict(data.items())

OrderedDumper.add_representer(OrderedDict, represent_ordered_mapping)

# end configure logging

parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/gpfs/home/unalg01/jepa/evals/')
    # default='/fsx-jepa/massran/submitit/')
parser.add_argument(
    '--exclude', type=str,
    help='nodes to exclude from training',
    default=None)
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--time', type=int, default=11520, #4300,
    help='time in minutes to run job')
parser.add_argument(
    '--log_dir', type=str, default="./logs",
    help='folder to save experiment logs')

class Trainer:

    def __init__(self, args_pretrain, log_dir=None, load_model=None):
        self.app = args_pretrain['app']
        self.args_pretrain = args_pretrain
        self.load_model = load_model
        self.log_dir = log_dir
        logger.info(f"In Trainer init: {args_pretrain}")

    def __call__(self):
        try:
            app = self.app
            params = self.args_pretrain
            load_model = self.load_model

            logger.info('loaded pretrain params...')
            pp = pprint.PrettyPrinter(indent=4)
            logger.info("Params:")
            logger.info(pp.pformat(params))
        
            # Launch app with loaded config
            resume_preempt = False if load_model is None else load_model
            app_main(app, args=params, resume_preempt=resume_preempt, log_dir=self.log_dir)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.exception("An error occurred during training.")
            raise e  # Re-raise the exception to ensure the job fails appropriately

    def checkpoint(self):
        fb_trainer = Trainer(self.args_pretrain, load_model=True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_app_with_parsed_args(
    args_for_pretrain,
    submitit_folder,
    partition,
    timeout= 11520, #4300,
    nodes=1,
    tasks_per_node=4,
    exclude_nodes=None,
    args_fname=None
):
    executor = submitit.AutoExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'),
        slurm_max_num_timeout=0) #20)
    executor.update_parameters(
        slurm_partition=partition,
        # slurm_mem_per_gpu='128G', 
        slurm_mem='256G',  #'192G',
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=6,
        gpus_per_node=tasks_per_node)

    if exclude_nodes is not None:
    # if args_exclude is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    logger.info(f"Executor parameters: {executor.parameters}")
    logger.info(f"tasks_per_node: {tasks_per_node}")
    logger.info(f"partition: {partition}")
    
    # Create log folder for the experiment
    log_dir = get_new_log_dir(args_for_pretrain[0]['logging']['folder'], prefix=f'mjepa_pretrain_distributed_', postfix='')

    if args_fname != None:
        yaml_params = None
        with open(args_fname, 'r') as y_file:
            yaml_params = yaml.load(y_file, Loader=OrderedLoader)
            
        logger.info('Writing params yaml to log dir ...')
        
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(yaml_params, f, Dumper=OrderedDumper, default_flow_style=False)
    
    jobs, trainers = [], []
    with executor.batch():
        for ap in args_for_pretrain:
            fb_trainer = Trainer(ap, log_dir)
            try:
                job = executor.submit(fb_trainer,)
                trainers.append(fb_trainer)
                jobs.append(job)
                # Cannot access job.job_id here
                logger.info("Job submitted (job ID not yet available).") 
            except Exception as e:
                logger.exception("Failed to submit job.")
                sys.exit(1)

    logger.info("All jobs submitted.")

    for job in jobs:
        logger.info(f"Submitted job with ID: {job.job_id}")
        # print(job.job_id)
        jobid_txt = os.path.join(log_dir, f"{job.job_id}.txt")  # Use job_id as the filename
        # Save the job ID in a file named after the job ID
        with open(jobid_txt, "w") as f:
            f.write(f"Job ID: {job.job_id}\n")

def launch():

    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    args = parser.parse_args()
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
        
        args.fname = None
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, 'r') as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)
            nodes = int(_params.get('nodes'))
            tasks_per_node = int(_params.get('tasks_per_node'))
            configs += [_params]
    logger.info(f'Loaded {len(configs)} config files')
    logger.info(f'Running all jobs with {nodes=} / {tasks_per_node=}')
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_app_with_parsed_args(
        args_for_pretrain=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude,
        args_fname=args.fname)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    try: 
        launch()
    except Exception as e:
        logger.exception("An error occurred in the launch process.")
        sys.exit(1)

