# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np
import random
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import torch.distributed as dist
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel

import torch.utils.tensorboard
import wandb

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys 
sys.path.append('/gpfs/home/unalg01/jepa')
sys.path.append('/home/gozde/medChangeDet/jepa')

from evals.video_classification_frozen.utils import plot_tsne
import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
    param_groups_lrd
)

from evals.video_classification_frozen.utils import (
    make_transforms,
    make_video_transforms,
    ClipAggregation,
    FrameAggregation
)

# logging.basicConfig(filename='my_log_file.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
# np.random.seed(_GLOBAL_SEED)
# torch.manual_seed(_GLOBAL_SEED)
# torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False, log_dir="./logs/evals"):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    print('Entry to main in eval')
    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    if ckp_fname is not None:
        pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    else:
        pretrained_path = None
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)
    in_chans = args_pretrain.get('in_channel_size', 3)
    frozen = args_pretrain.get('frozen', True)
    encoder_warmup = args_pretrain.get('encoder_warmup', 1)
    use_pos_embed = args_pretrain.get('use_pos_embed', False)
    clip_grad_encoder = args_pretrain.get('clip_grad_encoder',1.0)

    # -- DATA
    args_data = args_eval.get('data')
    # train_data_path = args_data.get('dataset_train', [])
    val_data_path = args_data.get('dataset_val', []) #[args_data.get('dataset_val')]
    #save_dir_path = args_data.get('root_path', '/gpfs/home/unalg01/jepa')
    #args_logging = args_eval.get('logging')
    #save_dir = args_logging.get('folder', '/gpfs/data/sodicksonlab/gozde/logs/')
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_clips = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    num_workers=args_data.get('num_workers',1)
    random_clip_sampling = args_data.get('random_clip_sampling', False)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    seed = args_opt.get('seed', _GLOBAL_SEED)
    classifier_depth = args_opt.get('classifier_depth', 1) 
    train_eval_freq = args_opt.get('train_log_iter_freq')
    val_eval_freq = args_opt.get('val_log_iter_freq')

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None) # tag: k400-16x8x3
    cls_checkpoint_path = args_eval.get('checkpoint_path')
    write_tag=args_eval.get('write_tag',"tsne")
    
    # TSNE ARGS
    max_samples = args_eval.get('max_samples', 100)

    # ----------------------------------------------------------------------- #
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        #device = torch.device('cuda:0')
        device = torch.device('cuda', rank % torch.cuda.device_count())  # safer for multi-GPU
        torch.cuda.set_device(device)

    if rank == 0:
        # wandb init
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="mjepa-project",
            
            entity="ituvisionlab",
            
            dir=log_dir,

            # track hyperparameters and run metadata
            config=args_eval,
            
            name="TSNE_" + os.path.basename(log_dir),
            
            group="tsne_eval"
            )
    else:
        run = None

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Only use deterministic algorithms if the function is available
        #if hasattr(torch, "use_deterministic_algorithms"):
        #    torch.use_deterministic_algorithms(True)

    seed_everything(seed)

    # Initialize model
    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        in_chans=in_chans,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=classifier_depth,
        num_classes=num_classes,
    ).to(device)
    print("Print the classifier")
    print(classifier)
    
    val_loader, val_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_clips=eval_num_clips,
        in_chans=in_chans,
        random_clip_sampling=random_clip_sampling,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        training=False)
       
    ipe = len(val_loader)
    logger.info(f'Val Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    classifier = DistributedDataParallel(classifier, static_graph=True, gradient_as_bucket_view=True)
    
    # Extract embeddings
    embeddings, labels = run_one_epoch(
        device=device,
        training=False,
        num_temporal_views=1,
        attend_across_segments=attend_across_segments,
        num_spatial_views=1,
        encoder=encoder,
        classifier=classifier,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        data_loader=val_loader,
        use_bfloat16=use_bfloat16,
        log_writer=None,
        eval_freq=train_eval_freq,
        rank=rank,
        max_samples=max_samples)
    
    # Visualize
    save_path = os.path.join(log_dir, f"{write_tag}_plot.png")
    plot_tsne(
        embeddings,
        labels,
        save_path=save_path,
        wandb_log=True,
        wandb_key="tsne/embeddings"
    )
    # #wandb logging
    # run.log({
    #     'val/tsne': wandb.Image(save_path, caption=f"t-SNE Embeddings - {write_tag}")
    #                 })

def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
    log_writer,
    eval_freq,
    rank,
    max_samples
):
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()

    ipe = len(data_loader)
    if eval_freq > ipe:
        eval_freq = 1

    logits = []
    logit_labels = []
    samples_count = 0

    for itr, data in enumerate(data_loader):

        # If we've already collected enough samples, break early
        if samples_count >= max_samples:
            break

        print(f"[Batch {itr+1}] Collected Samples: {samples_count} / {max_samples}")

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Move input data to device
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # spatial views
                for di in data[0]  # temporal views
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = labels.shape[0]

            with torch.no_grad():
                outputs = encoder(clips, clip_indices)

            # Convert outputs to CPU numpy arrays
            outputs = [out.cpu().numpy() for out in outputs]  # list of arrays, typically len 1

            # Extend the embeddings
            logits.extend(outputs)
            logit_labels.append(labels.cpu().numpy())

            samples_count += batch_size

        torch.cuda.empty_cache()

    # After loop ends
    if len(logits) == 0:
        raise RuntimeError("No logits collected, check your data loader or encoder.")

    logits = np.concatenate(logits, axis=0)

    # Global average pooling over L dimension if needed
    if logits.ndim == 3:  # (B, L, D)
        logits = np.mean(logits, axis=1)  # Pool across L

    print("Collected feature embeddings shape:", logits.shape)

    logit_labels = np.concatenate(logit_labels, axis=0)

    # If more than max_samples were collected in the last batch, truncate
    if logits.shape[0] > max_samples:
        logits = logits[:max_samples]
#        logit_labels = logit_labels[:max_samples]

    return logits, logit_labels


def load_checkpoint(
    device,
    r_path,
    classifier,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['classifier']
        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    print("Print the pretrained model")
    print(checkpoint.keys())
    #print(checkpoint['classifier'])
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_clips=8,
    in_chans=3,
    random_clip_sampling=False,
    auto_augment=False,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(1.0,1.0), #(0.75, 4/3),
    random_resize_scale=(0.9, 1.0),
    rot_degree=10,
    intensity_gamma=0.2,
    random_bias=0.2,
    random_noise=0.025,
    num_workers=4,
    subset_file=None
):
    # Make Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(1.0,1.0),
        random_resize_scale=(0.9, 1.0),
        rot_degree=10,
        reprob=0,
        auto_augment=auto_augment,
        motion_shift=False,
        crop_size=resolution,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise,
        in_chans=in_chans
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_clips,
        in_chans=in_chans,
        crop_size=resolution,
        random_clip_sampling=random_clip_sampling, 
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
        training=training)
    return data_loader, data_sampler


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    in_chans=3,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        in_chans= in_chans,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
