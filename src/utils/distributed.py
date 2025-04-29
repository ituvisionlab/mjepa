# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import platform

import torch
import torch.distributed as dist

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

from logging import getLogger

logger = getLogger()


def init_distributed(port=37123, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        
        hostname = platform.node()
        backend_engine = "nccl"
        if hostname == "panther": # nccl backend doesn't work on panther machine for now
            backend_engine = "gloo"
            
        dist.init_process_group(
            backend=backend_engine, # nccl
            world_size=world_size,
            rank=rank,
            init_method='env://'
        )
        dist.barrier()
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'Rank: {rank}. Distributed training not available {e}')

    print(f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
    print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

    return world_size, rank

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
# #obsolete
# def all_reduce_mean(tensor):
#     """ Averages a tensor across all distributed processes. """
#     if dist.is_available() and dist.is_initialized():
#         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#         tensor /= dist.get_world_size()
#     return tensor



def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0

def gather_all_tensors(tensor):
    """
    Gathers tensor data from all processes and concatenates.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def compute_distributed_auc(all_outputs, all_labels, num_classes):
    """
    Computes AUC score in both single and multi-GPU settings.
    
    Args:
        all_outputs (List[Tensor]): List of model outputs per batch (after softmax).
        all_labels (List[Tensor]): List of ground truth labels per batch.
        num_classes (int): Number of target classes.

    Returns:
        float: AUC score (macro averaged in multi-class), NaN-safe.
    """
    # Combine all batches
    all_outputs_tensor = torch.cat(all_outputs, dim=0).contiguous()
    all_labels_tensor = torch.cat(all_labels, dim=0).contiguous()

    # Detect distributed setup
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    if is_dist:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        all_outputs_tensor = all_outputs_tensor.to('cuda')
        all_labels_tensor = all_labels_tensor.to('cuda')

        gathered_outputs = [torch.zeros_like(all_outputs_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered_outputs, all_outputs_tensor)
        torch.distributed.all_gather(gathered_labels, all_labels_tensor)

        all_outputs_tensor = torch.cat(gathered_outputs, dim=0).cpu()
        all_labels_tensor = torch.cat(gathered_labels, dim=0).cpu()
    else:
        all_outputs_tensor = all_outputs_tensor.cpu()
        all_labels_tensor = all_labels_tensor.cpu()

    # Convert to numpy for sklearn
    labels_np = all_labels_tensor.numpy()
    outputs_np = all_outputs_tensor.numpy()

    try:
        if num_classes == 2:
            # Binary classification (use class 1 probs)
            if len(np.unique(labels_np)) > 1:
                auc_final = roc_auc_score(labels_np, outputs_np[:, 1])
            else:
                auc_final = float('nan')
        else:
            # Multi-class classification
            labels_bin = label_binarize(labels_np, classes=np.arange(num_classes))
            if len(np.unique(labels_np)) > 1:
                auc_final = roc_auc_score(
                    labels_bin,
                    outputs_np,
                    average='macro',
                    multi_class='ovr'
                )
            else:
                auc_final = float('nan')
    except Exception as e:
        print(f"[DEBUG AUC ERROR] {e}")
        auc_final = float('nan')

    return auc_final
