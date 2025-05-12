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
import logging
import os
import pprint
import sys
import time
import yaml
from collections import OrderedDict

import submitit

from evals.scaffold import main as eval_main
from app.vjepa.utils import get_new_log_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO) # ,filename='main_log_file.log'
logger = logging.getLogger()

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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/fsx-jepa/massran/submitit/')
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
    '--reservation', type=str,
    help='cluster reservation to submit jobs on')
parser.add_argument(
    '--time', type=int, default= 4300,
    help='time in minutes to run job')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='number of nodes')
parser.add_argument(
    '--log_dir', type=str, default="./logs",
    help='folder to save experiment logs')

class Trainer:

    def __init__(self, args_eval=None, resume_preempt=None, log_dir=None):
        self.eval_name = args_eval['eval_name']
        self.args_eval = args_eval
        self.log_dir = log_dir
        self.resume_preempt = resume_preempt

    def __call__(self):
        eval_name = self.eval_name
        args_eval = self.args_eval
        resume_preempt = self.resume_preempt

        logger.info('loaded eval params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args_eval)

        eval_main(
            eval_name,
            args_eval=args_eval,
            resume_preempt=resume_preempt,
            log_dir=self.log_dir)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_eval, True, self.log_dir)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_evals_with_parsed_args(
    args_for_evals,
    submitit_folder,
    partition='a100_short', #'learnlab,learnfair',
    reservation=None,
    timeout= 4300,
    nodes=1,
    tasks_per_node=4,
    delay_seconds=10,
    exclude_nodes=None,
    args_fname=None
):
    # # 1. Initialize submitit JobEnvironment
    # env = submitit.JobEnvironment()
    
    # # 2. Set the environment variables for distributed training
    # os.environ["RANK"] = str(env.global_rank)
    # os.environ["WORLD_SIZE"] = str(env.num_tasks)
    # os.environ["LOCAL_RANK"] = str(env.local_rank)
    # os.environ["MASTER_ADDR"] = env.hostnames[0]  # Rank 0's host
    # os.environ["MASTER_PORT"] = "29500"  # Choose a free port

    # logger.info(f'Initialized distributed environment: RANK={os.environ["RANK"]}, '
    #             f'WORLD_SIZE={os.environ["WORLD_SIZE"]}, MASTER_ADDR={os.environ["MASTER_ADDR"]}, '
    #             f'MASTER_PORT={os.environ["MASTER_PORT"]}')

    if not isinstance(args_for_evals, list):
        logger.info(f'Passed in eval-args of type {type(args_for_evals)}')
        args_for_evals = [args_for_evals]

    time.sleep(delay_seconds)
    logger.info('Launching evaluations in separate jobs...')
    #executor = submitit.AutoExecutor(
    executor =submitit.SlurmExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'))
        #slurm_max_num_timeout=0) #20)
    if reservation:  # Add reservation only if provided
        executor.update_parameters(
           slurm_partition=partition,
           # slurm_reservation=reservation,
           # slurm_mem_per_gpu='128G', 
           slurm_mem='128G',  #'192G',
           timeout_min=timeout,
           nodes=nodes,
           ntasks_per_node=tasks_per_node,
           cpus_per_task= 4, #6, #for num_workers=4  
           gpus_per_node=tasks_per_node,
           slurm_additional_parameters={
            'reservation': reservation,
            'output': "/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log"
            }
        )
    else:
        slurm_params = {
            'partition': partition,
            'mem': '128G',  # Adjust memory per your needs
            'time': timeout,
            'nodes': nodes,
            'ntasks_per_node': tasks_per_node,
            'cpus_per_task': 4, #6,
            'gpus_per_node': tasks_per_node,
            'additional_parameters': {
                'output': "/gpfs/data/sodicksonlab/gozde/slurm/slurmDistrib_%j.log"
            }
        }   
        executor.update_parameters(**slurm_params)

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

     # Create log folder for the experiment
    log_dir = get_new_log_dir(args_for_evals[0]['logging']['folder'], prefix=f'{args_for_evals[0]["write_tag"]}_eval_distributed_', postfix='')
    # Load YAML params and write to log
    if args_fname != None:
        yaml_params = None
        with open(args_fname, 'r') as y_file:
            yaml_params = yaml.load(y_file, Loader=OrderedLoader)
            
        logger.info('Writing params yaml to log dir ...')
        
        dump = os.path.join(log_dir, 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(yaml_params, f, Dumper=OrderedDumper, default_flow_style=False)
    # Submit jobs
    jobs, trainers = [], []
    with executor.batch():
        for ae in args_for_evals:
            fb_trainer = Trainer(ae, log_dir=log_dir)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        logger.info(f'Launched eval job with id {job.job_id}')
        jobid_txt = os.path.join(log_dir, f"{job.job_id}.txt")  # Use job_id as the filename
        # Save the job ID in a file named after the job ID
        with open(jobid_txt, "w") as f:
            f.write(f"Job ID: {job.job_id}\n")

def launch_evals():

    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
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
    launch_evals_with_parsed_args(
        args_for_evals=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        reservation=args.reservation,
        timeout=args.time,
        nodes=args.nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude,
        args_fname=args.fname)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    print("Entered main_distributed.py")
    launch_evals()
