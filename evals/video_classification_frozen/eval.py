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
import platform

import numpy as np
import random
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import torch.distributed as dist
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel

#import torch.utils.tensorboard
import argparse
import wandb
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import sys 
sys.path.append('/gpfs/home/unalg01/jepa')
sys.path.append('/home/gozde/medChangeDet/jepa')

import math

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.models.attentive_pooler import AttentionPooling

from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import (
    init_distributed,
    init_distributed_mode,
    compute_distributed_auc,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
    param_groups_lrd
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)
from src.utils.tensors import trunc_normal_

from contextlib import nullcontext

# logging.basicConfig(filename='my_log_file.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

checkpoint_freq = 1
save_ckpt_epoch_freq = 10
_GLOBAL_SEED = 0
#np.random.seed(_GLOBAL_SEED)
#torch.manual_seed(_GLOBAL_SEED)
#torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False, log_dir="./logs/evals"):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    print('Entry to main in eval')
    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
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
    in_chans = args_pretrain.get('in_channel_size', 1)
    frozen = args_pretrain.get('frozen', True)
    encoder_warmup = args_pretrain.get('encoder_warmup', 1)
    use_pos_embed = args_pretrain.get('use_pos_embed', False)
    clip_grad_encoder = args_pretrain.get('clip_grad_encoder',1.0)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)

    # -- DATA
    args_data = args_eval.get('data')
    # train_data_path = [args_data.get('dataset_train')]
    train_data_path = args_data.get('dataset_train', [])
    val_data_path = args_data.get('dataset_val', []) #[args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'MRIDataset')
    num_classes = args_data.get('num_classes')
    eval_num_clips = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_in_chans = args_data.get('eval_in_channel_size', 1)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    num_workers=args_data.get('num_workers',1)
    random_clip_sampling = args_data.get('random_clip_sampling', False)
    # -- DATA Augmentation
    args_data_aug = args_eval.get('data_aug')
    auto_augment = args_data_aug.get('auto_augment', False)
    random_noise = args_data_aug.get('random_noise', 0.025)
    random_bias = args_data_aug.get('random_bias', 0.2)
    intensity_gamma = args_data_aug.get('intensity_gamma', 0.2)
    rot_degree = args_data_aug.get('rotation_degree', 0.0)
    random_resize_aspect_ratio = args_data_aug.get('random_resize_aspect_ratio', [1, 1])
    random_resize_scale = args_data_aug.get('random_resize_scale', [0.9, 1.0])
    random_horizontal_flip = args_data_aug.get('random_horizontal_flip', True)
    
    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = float(args_opt.get('weight_decay'))
    final_wd = float(args_opt.get('final_weight_decay'))
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    seed = args_opt.get('seed', _GLOBAL_SEED)
    train_eval_freq = args_opt.get('train_log_iter_freq')
    val_eval_freq = args_opt.get('val_log_iter_freq')
    clip_grad_classifier = args_opt.get('clip_grad_classifier',1.0)
    betas = args_opt.get('betas', (0.9, 0.999))
    eps = args_opt.get('eps', 1.e-6) #1e-8 or 1e-7?
    layer_decay = args_opt.get('layer_decay', None)
    classifier_depth = args_opt.get('classifier_depth', 1) 
    dropout = args_opt.get('dropout', None)
    drop_rate = args_opt.get('drop_rate', 0.0)
    attn_drop_rate = args_opt.get('attn_drop_rate', 0.0) 
    accumulation_steps = args_opt.get('grad_accum_steps', 2)  # Define the number of steps before updating the optimizer 

   
    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None) # tag: k400-16x8x3
    cls_checkpoint_path = args_eval.get('checkpoint_path')

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        #device = torch.device('cuda:0')
        device = torch.device('cuda', rank % torch.cuda.device_count())  # safer for multi-GPU
        torch.cuda.set_device(device)

    # -- log/checkpointing paths   
    if log_dir != None:
        model_folder = os.path.join(log_dir, "model_ckpt")
        csv_folder = os.path.join(log_dir, "csv_logs")
        tb_folder = os.path.join(log_dir, "tensorboard")
        
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(tb_folder, exist_ok=True)
        
        csv_log_file = os.path.join(csv_folder, f'{eval_tag}_r{rank}.csv')
        
        
        # Model checkpoint folders
        
        latest_model_folder = os.path.join(model_folder, "latest-model")
        best_model_folder = os.path.join(model_folder, "best-model")
        periodic_model_folder = os.path.join(model_folder, "periodic-model")
        
        os.makedirs(latest_model_folder, exist_ok=True)
        os.makedirs(best_model_folder, exist_ok=True)
        os.makedirs(periodic_model_folder, exist_ok=True)
        
        latest_path = os.path.join(latest_model_folder, f'{eval_tag}-latest.pth.tar')
        latest_info_path = os.path.join(latest_model_folder, f'latest-info.txt')
        
        best_path = os.path.join(best_model_folder, f'{eval_tag}-best.pth.tar')
        best_info_path = os.path.join(best_model_folder, f'best-info.txt')
        
        
        # Tensorboard logging
        #tb_rank_folder = os.path.join(tb_folder, f"{eval_tag}_rank_{rank}")
        # os.makedirs(tb_rank_folder, exist_ok=True)
        log_writer = None #torch.utils.tensorboard.SummaryWriter(tb_rank_folder)
        
        # -- make csv_logger
        csv_logger = CSVLogger(csv_log_file,
                                ('%d', 'epoch'),
                                ('%.5f', 'train acc'),
                                ('%.5f', 'val acc'),
                                ('%.5f', 'train loss'),
                                ('%.5f', 'val loss'),
                                ('%.5f', 'train recall'),
                                ('%.5f', 'val recall'),
                                ('%.5f', 'train precision'),
                                ('%.5f', 'val precision'),
                                ('%.5f', 'train f1'),
                                ('%.5f', 'val f1'),
                                ('%.5f', 'val AUC'))
        
        if rank == 0:
            # wandb init
            hostname = platform.node()
            entity_name = "mgulsen2020-wandb" if hostname == "panther" else "ituvisionlab"
            
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="mjepa-project",
                
                entity=entity_name,
                
                dir=log_dir,

                # track hyperparameters and run metadata
                config=args_eval,
                
                name=os.path.basename(log_dir)
                
                # group="mjepa-DDP"
                )
        else:
            run = None
    else:
        model_folder = None
        csv_folder = None
        tb_folder = None
        csv_log_file = None
        latest_model_folder = None
        best_model_folder = None
        periodic_model_folder = None
        latest_path = None
        latest_info_path = None
        best_path = None
        best_info_path = None
        tb_rank_folder = None
        log_writer = None
        csv_logger = None
        run = None

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
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
    # -- pretrained encoder (frozen) or unfrozen
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
        use_sdpa=use_sdpa,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
        )
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments,
            use_pos_embed=use_pos_embed
        ).to(device)

    # for multi-channel inputs
    attn_pooler = AttentionPooling(embed_dim=encoder.model.embed_dim).to(device)
    print("Print the attention pooler")
    print(attn_pooler)

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=classifier_depth,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    print("Print the classifier")
    print(classifier)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Classifier number of parameters: {count_parameters(classifier)}')
    
    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_clips=eval_num_clips, #if attend_across_segments else 1,
        num_views_per_segment=1,
        in_chans=eval_in_chans,
        random_clip_sampling=random_clip_sampling,
        auto_augment=auto_augment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        rot_degree=rot_degree,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        training=True)
    val_loader, val_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_clips=eval_num_clips,
        in_chans=eval_in_chans,
        random_clip_sampling=random_clip_sampling,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        rot_degree=rot_degree,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')
    
    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        encoder=encoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
        frozen=frozen,
        betas=betas,
        eps=eps,
        layer_decay=layer_decay)
    classifier = DistributedDataParallel(classifier, static_graph=True, gradient_as_bucket_view=True)

    if not frozen:
        encoder = DistributedDataParallel(encoder, static_graph=True, gradient_as_bucket_view=True) #GU_Debug

    if frozen:
        encoder.eval()    
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        encoder.train()    
        for name, param in encoder.named_parameters():
            if "pos_embed" in name:
                param.requires_grad = False  # Keep pos_embed frozen even when training
            else:
                param.requires_grad = True

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=cls_checkpoint_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, train_acc, val_acc, path, info_path):
        save_dict = {
            'encoder': encoder.state_dict(),  # Save encoder state
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, path) #rather than to save on latest_path pretrained classifier
            with open(info_path, "w") as info_f:
                info_f.write(f"Model path: {path},\nEpoch: {epoch+1},\ntrain acc: {train_acc}, val acc: {val_acc}, lr: {lr}")
                
    epoch_accs = []
    epoch_val_accs = []
    encoder_frozen = True
    
    if pretrained_path == None:
        encoder_warmup = 0

    # if rank == 0:
    #     sanity_check(encoder, classifier, val_loader, device, num_classes)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            logger.info('Epoch %d' % (epoch))

        if not frozen:
            if epoch >= encoder_warmup:
                encoder_frozen = False

        train_acc, train_loss, train_recall, train_precision, train_f1, auc_score = run_one_epoch(
            device=device,
            training=True,
            num_temporal_views=eval_num_clips, #if attend_across_segments else 1,
            attend_across_segments=attend_across_segments,
            num_spatial_views=1,
            encoder=encoder,
            classifier=classifier,
            attn_pooler=attn_pooler,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            data_sampler=train_sampler,
            use_bfloat16=use_bfloat16,
            frozen=encoder_frozen,
            log_writer=log_writer,
            epoch=epoch,
            eval_freq=train_eval_freq,
            rank=rank,
            run=run,
            num_classes=num_classes,
            warmup=warmup,
            clip_grad_encoder=clip_grad_encoder,
            clip_grad_classifier=clip_grad_classifier,
            accumulation_steps=accumulation_steps,
            log_dir=log_dir,
            eval_in_chans=eval_in_chans,)

        val_acc, val_loss, val_recall, val_precision, val_f1, auc_score = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_clips,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            classifier=classifier,
            attn_pooler=attn_pooler,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            data_sampler=val_sampler,
            use_bfloat16=use_bfloat16,
            frozen=encoder_frozen,
            log_writer=log_writer,
            epoch=epoch,
            eval_freq=val_eval_freq,
            rank=rank,
            run=run,
            num_classes=num_classes,
            warmup=warmup,
            clip_grad_encoder=clip_grad_encoder,
            clip_grad_classifier=clip_grad_classifier,
            accumulation_steps=accumulation_steps,
            log_dir=log_dir,
            eval_in_chans=eval_in_chans,)

        #GU_ DEBUG
        #if not math.isnan(auc_score):
        #    print(f"Val AUC: {auc_score:.4f} at epoch: {epoch}")
        auc_score=0
        if rank == 0:
            logger.info('[%5d] train: %.3f%% test: %.3f%% AUC: %.3f' % (epoch, train_acc, val_acc, auc_score))
        
        # if rank == 0:
        if csv_logger != None:
            csv_logger.log(epoch, train_acc, val_acc, train_loss, val_loss, train_recall, val_recall, train_precision, val_precision, train_f1, val_f1, auc_score)
        
        #if (epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1)) and log_dir != None:
        if log_dir != None: # at the end of every epoch       
            if not os.path.exists(latest_path):
                save_checkpoint(epoch, train_acc, val_acc, latest_path, latest_info_path)
            else:
                if len(epoch_accs) > 0:
                    if val_acc > max(epoch_val_accs) and epoch > 4: #if train_acc > max(epoch_accs) and epoch > 4:
                        save_checkpoint(epoch, train_acc, val_acc, best_path, best_info_path)
                    else:
                        save_checkpoint(epoch, train_acc, val_acc, latest_path, latest_info_path)
                    if epoch% save_ckpt_epoch_freq ==0:
                        periodic_path = os.path.join(periodic_model_folder, f'{eval_tag}-periodic-epoch-{epoch}.pth.tar')
                        periodic_info_path = os.path.join(periodic_model_folder, f'periodic-info-epoch-{epoch}.txt')
                        save_checkpoint(epoch, train_acc, val_acc, periodic_path, periodic_info_path)
        if epoch >= 4:
            epoch_accs.append(train_acc)
            epoch_val_accs.append(val_acc)
            
    if run != None:
        run.finish()


def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    attn_pooler,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    data_sampler,
    use_bfloat16,
    frozen,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
    log_writer,
    epoch,
    eval_freq,
    rank,
    run,
    num_classes,
    warmup,
    clip_grad_encoder,
    clip_grad_classifier,
    accumulation_steps,
    log_dir,
    eval_in_chans,
):

    classifier.train(mode=training)
    if frozen: 
        encoder.eval()
    else:
        encoder.train(mode=training) 
    
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    auroc_meter = AverageMeter()
    recall_meter = AverageMeter()
    #specificity_meter = AverageMeter()
    f1_meter = AverageMeter()
    precision_meter = AverageMeter()
    ipe = len(data_loader)
    if eval_freq > ipe:
        eval_freq = 1
    auc_score = 0 #AUC Calculations are cancelled!!!! SET TO ZERO!!!

    loader = iter(data_loader)
    data_sampler.set_epoch(epoch)
    
    loss = None
    all_outputs = []
    all_labels = []

    # for itr, data in enumerate(data_loader):
    for itr in range(ipe):
        try:
            data = next(loader)
        except Exception:
            logger.info('Exhausted data loaders. Refreshing...')
            torch.cuda.empty_cache()
                
            loader = iter(data_loader) #resets the loader iterator again
            data = next(loader)
   
        new_lr = None
        new_wd = None
        if training:
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
        #with torch.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU: move frames to GPU
            labels = data[1].to(device)
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            batch_size = len(labels)
            if eval_in_chans > 1:
                # MULTI-CONTRAST INPUT (e.g., prostate MRI)
                full_clip = data[0][0][0]  # shape: [B, 3, T, H, W]

                # Split channels : contrasts[i]: B, 1, T, H, W
                contrasts = [full_clip[:, i:i+1].to(device, non_blocking=True) for i in range(eval_in_chans)]

                # Wrap as [[ [Tensor] ]] for each contrast to match expected encoder input structure
                inputs_per_contrast = [[[contrast]] for contrast in contrasts]
            else:
                # SINGLE-CHANNEL INPUT
                clips = [
                    [dij.to(device, non_blocking=True) for dij in di] # iterate over spatial views of clip
                    for di in data[0] # iterate over temporal index of clip
                ]


            # clips list: len = no_of_clips 
            # e.g. clips[0][0].shape -> torch.Size([4, 3, 16, 224, 224]): B x C x T X W X H
            # clips[1][0].shape ""
            # Forward and prediction
            outputs = None
            if eval_in_chans > 1:
                # Multi-contrast evaluation
                context = torch.no_grad() if frozen or not training else torch.enable_grad()
                with context:
                    encoded_contrasts = [ #encoded_contrasts[i=0:2].shape: B, N, D
                        encoder(inp, clip_indices)[0]
                        for inp in inputs_per_contrast if inp is not None
                    ]

                    # encoded_contrasts = [
                    #     encoder(inp, clip_indices)
                    #     for inp in inputs_per_contrast if inp is not None
                    # ]  # list of [B, N, D]  
            
                outputs = [attn_pooler(encoded_contrasts)]  # list of one [B, N, D]
            else:
                # Original single-channel pipeline
                if not frozen:
                    if training:
                        outputs = encoder(clips, clip_indices)
                    else:
                        with torch.no_grad():
                            outputs = encoder(clips, clip_indices)
                else:
                    with torch.no_grad():
                        outputs = encoder(clips, clip_indices)

            # if not frozen:
            #     if training:
            #         outputs = encoder(clips, clip_indices)
            #     else:
            #         with torch.no_grad():
            #             outputs = encoder(clips, clip_indices)

            # with torch.no_grad():
            #     if frozen:
            #         outputs = encoder(clips, clip_indices) #outputs[0].shape= torch.Size([4, 3136, 1024])
                
                if not training:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
            if training:
                
                # print(f"Outputs = Clip embeddings require grad: {outputs[0].requires_grad}")
                # print(f"Outputs = Clip embeddings require grad: {outputs[0][0].requires_grad}") #for not attend_across_segments
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]
        # outputs tensor shape: Batchsize x num_classes
        #GU_Debug
        # print(f"Classifier Outputs require grad: {outputs[0].requires_grad}")

        # Compute loss
        if attend_across_segments:
            loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        else:
            loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
        with torch.no_grad():
            if attend_across_segments:
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
            top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)
            
            # Compute additional metrics per batch
            preds = outputs.max(dim=1).indices
           # Use macro average for multiclass classification
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro') # average over all classes
            # recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average=None) # per class average
            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
           
           # cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
           # tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
           # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Append current batch results for AUC calculations during val
            if not training:
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())
    
            recall = float(AllReduce.apply(torch.tensor(recall, device='cuda')))
            precision = float(AllReduce.apply(torch.tensor(precision, device='cuda')))
            #specificity = float(AllReduce.apply(torch.tensor(specificity, device='cuda')))
            f1 = float(AllReduce.apply(torch.tensor(f1, device='cuda')))
            recall_meter.update(recall)
            precision_meter.update(precision)
            # specificity_meter.update(specificity)
            f1_meter.update(f1)

        # GU_debug: Check if the encoder is frozen
        # encoder_frozen = True
        # for name, param in encoder.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         encoder_frozen = False
        #         print(f"Gradient found for encoder parameter: {name}, norm: {param.grad.norm().item()}")
        #         break

        # if encoder_frozen:
        #     print("Encoder is fully frozen. No gradients are propagated.")
        # else:
        #     print("Encoder is not frozen. Gradients are propagating to some parameters.")
        #end_debug
        
        loss = loss / accumulation_steps
        torch.cuda.synchronize()
        
        if not torch.isfinite(loss):
            logger.warning(f"[Rank {rank}] Non-finite loss detected: {loss}")
            loss = torch.tensor(0.0, device=loss.device)
        loss = AllReduce.apply(loss)  # Average loss across GPUs  

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                if (itr + 1) % accumulation_steps == 0:  # Only unscale when we're going to step
                    scaler.unscale_(optimizer)
                if epoch > warmup:
                    if (clip_grad_classifier is not None):
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad_classifier)
                    if not frozen and (clip_grad_encoder is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) # newly added 1/19/2025
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if epoch > warmup:
                    if (clip_grad_classifier is not None):
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad_classifier)
                    if not frozen and (clip_grad_encoder is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) # newly added 1/19/2025
                optimizer.step()
            # # GU_Debug
            # for name, param in encoder.named_parameters():
            #     if param.grad is None:
            #         print(f"No gradient for: {name}")
            #     else:
            #         print(f"Gradient exists for: {name} | Norm: {param.grad.norm().item()}")
            # # GU_Debug
            # for name, param in classifier.named_parameters():
            #     if param.grad is None:
            #         print(f"Classifier: No gradient for: {name}")
            #     else:
            #         print(f"Classifier Gradient exists for: {name} | Norm: {param.grad.norm().item()}")
            
            if (itr + 1) % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)  # Efficient way to clear gradients
            #optimizer.zero_grad(set_to_none=True)

        # Tensorboard logging cancelled: log_writer set to None
        if log_writer != None:
            if training and itr % eval_freq == 0:
                log_writer.add_scalar('train/acc', top1_meter.avg, (epoch * ipe) + itr)
                log_writer.add_scalar('train/loss', loss, (epoch * ipe) + itr)
                log_writer.add_scalar('train/mem', torch.cuda.max_memory_allocated() / 1024.**2, (epoch * ipe) + itr)
            
            if not training and itr % eval_freq == 0:
                log_writer.add_scalar('val/acc', top1_meter.avg, (epoch * ipe) + itr)
                log_writer.add_scalar('val/loss', loss, (epoch * ipe) + itr)
                log_writer.add_scalar('val/mem', torch.cuda.max_memory_allocated() / 1024.**2, (epoch * ipe) + itr)
            
            log_writer.flush()
        
        # Wandb logging
        if run != None and rank == 0:
            if training and itr % eval_freq == 0:
                run.log({
                        'train/acc': top1_meter.avg,
                        'train/loss': loss,
               #         'train/auroc': auroc_meter.avg,
                        'train/recall': recall_meter.avg,
                        'train/precision': precision_meter.avg,
                        # 'train/specificity': specificity_meter.avg,
                        'train/f1': f1_meter.avg,
                        'train/mem': torch.cuda.max_memory_allocated() / 1024.**2,
                        'train/lr': new_lr,
                        'train/wd': new_wd
                    })
            
            # Wandb logging
            if not training and itr % eval_freq == 0:
                run.log({
                        'val/acc': top1_meter.avg,
                        'val/loss': loss,
               #         'val/auroc': auroc_meter.avg,
                        'val/recall': recall_meter.avg,
                        'val/precision': precision_meter.avg,
                       # 'val/specificity': specificity_meter.avg,
                        'val/f1': f1_meter.avg,
                        'val/mem': torch.cuda.max_memory_allocated() / 1024.**2
                    })
        if itr % 5 == 0 and rank == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))

    #end of one epoch
    # if rank == 0 and not training:
    #     auc_score = compute_distributed_auc(
    #         all_outputs, all_labels, num_classes,
    #         save_path=os.path.join(log_dir, "auc_debug") if log_dir else None,
    #         step=epoch
    #     )
    # else:
    #     auc_score = 0 #float('nan')

    # # log AUC after one epoch is completed for validation set
    # if run is not None and rank == 0 and not training:
    #     if isinstance(auc_score, float) and not np.isnan(auc_score):
    #         run.log({'val/auc': auc_score})
    #     else:
    #         logger.warning(f"[Rank {rank}] Skipping AUC log due to NaN")
        
    torch.cuda.empty_cache()
        
    return top1_meter.avg, loss, recall, precision, f1, auc_score

@torch.no_grad()
def sanity_check(encoder, classifier, val_loader, device, num_classes):
    print("\n Running Sanity Check on Evaluation Pipeline...")

    encoder.eval()
    classifier.eval()

    loader_iter = iter(val_loader)
    data = next(loader_iter)

    # Format clips
    clips = [
        [dij.to(device) for dij in di] for di in data[0]
    ]
    labels = data[1].to(device)
    clip_indices = [d.to(device) for d in data[2]]

    # Encoder forward
    outputs = encoder(clips, clip_indices)
    print(f"[ENCODER] output[0] shape: {outputs[0].shape}")

    # Classifier forward
    if isinstance(outputs[0], torch.Tensor):  # attend_across_segments
        preds = [classifier(o) for o in outputs]
        probs = sum([F.softmax(p, dim=1) for p in preds]) / len(preds)
    else:
        preds = [[classifier(ost) for ost in os] for os in outputs]
        probs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in preds]) / len(preds) / len(preds[0])

    print(f"[CLASSIFIER] probs shape: {probs.shape}")

    # Loss check
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(probs, labels)
    print(f"[LOSS] {loss.item()}")

    # Prediction sanity
    print(f"[LABELS] shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}")
    print(f"[PRED]  top-1 indices: {probs.argmax(dim=1)[:8]}")
    print("Sanity check passed.\n")

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
    checkpoint_key='encoder' #'target_encoder'
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
    dataset_type='MRIDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_clips=8,
    in_chans=1,
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
    in_chans=1,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='encoder', #'target_encoder',
    drop_rate=0.0,
    attn_drop_rate=0.0
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
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )

    if pretrained is not None:
        encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    encoder.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Encoder number of parameters: {count_parameters(encoder)}')

    return encoder

def init_opt(
    classifier,
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    frozen=True,
    betas=(0.9, 0.999),
    eps=1e-8,
    layer_decay=None
):
    param_groups = [
        {
            'params': [p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)],
            'names': [n for n, p in classifier.named_parameters()
                      if ('bias' not in n) and (len(p.shape) != 1)],
        }, 
        {
            'params': [p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)],
            'names': [n for n, p in classifier.named_parameters()
                      if ('bias' in n) or (len(p.shape) == 1)],
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    
    if not frozen:
        
        if layer_decay is None:
            param_groups.extend([
                {
                    'params': [p for n, p in encoder.named_parameters()
                               if ('bias' not in n) and (len(p.shape) != 1)],
                    'names': [n for n, p in encoder.named_parameters()
                              if ('bias' not in n) and (len(p.shape) != 1)],
                },
                {
                    'params': [p for n, p in encoder.named_parameters()
                               if ('bias' in n) or (len(p.shape) == 1)],
                    'names': [n for n, p in encoder.named_parameters()
                              if ('bias' in n) or (len(p.shape) == 1)],
                    'WD_exclude': True,
                    'weight_decay': 0,
                }
            ])
        else:
            encoder_param_groups = param_groups_lrd(encoder.model, wd, 
                                                    no_weight_decay_list={'pos_embed', 'cls_token', 'dist_token', 'bias'},
                                                    layer_decay=layer_decay)
            param_groups.extend(encoder_param_groups)

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
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
    
    # **Debugging: Print parameter names, shapes, and requires_grad**
    # print("\nOptimizer Parameter Groups Debug Info:")
    # for group_idx, group in enumerate(optimizer.param_groups):
    #     print(f"\n--- Parameter Group {group_idx + 1} ---")
    #     for name, param in zip(group.get('names', []), group['params']):
    #         print(f"Name: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")

    return optimizer, scaler, scheduler, wd_scheduler
