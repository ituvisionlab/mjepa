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
import wandb.errors

from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix
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
#from src.utils.tensors import trunc_normal_
#from contextlib import nullcontext

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
    eval_frame_step = args_pretrain.get('frame_step', 1)
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
    threshold_isotropy = args_data_aug.get('threshold_isotropy', 1.4)
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
    warmup = args_opt.get('warmup',0)
    lr_schedule_factor = args_opt.get('lr_schedule_factor',1.0)
    use_bfloat16 = args_opt.get('use_bfloat16')
    attn_pooler_flag = args_opt.get('attn_pooler_flag',False)
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
                                ('%.5f', 'val acc'),
                                ('%.5f', 'val loss'),
                                ('%.5f', 'val recall'),
                                ('%.5f', 'val precision'),
                                ('%.5f', 'val f1'),
                                ('%.5f', 'val AUC'))
        
        if rank == 0:
            try: # wandb init
                logger.info(f"[RANK {rank}] Attempting to init wandb (online mode)...")
                hostname = platform.node()
                # entity_name = "mgulsen2020-wandb" if hostname == "panther" else "ituvisionlab"
                entity_name = "ituvisionlab"
                run = wandb.init(
                    project="mjepa-project", # set the wandb project where this run will be logged
                    entity=entity_name,
                    dir=log_dir,
                    config=args_eval, # track hyperparameters and run metadata
                    name=os.path.basename(log_dir),
                    settings=wandb.Settings(init_timeout=60) # 1 minute timeout
                    )
                logger.info(f"[RANK {rank}] wandb initialized (online).")
            except wandb.errors.CommError as e:
                logger.warning(f"[RANK {rank}] wandb.init() failed: {e}. Switching to offline mode.")
                os.environ["WANDB_MODE"] = "offline"
                run = wandb.init(
                    project="mjepa-project",
                    entity=entity_name,
                    dir=log_dir,
                    config=args_eval,
                    name=os.path.basename(log_dir),
                    mode="offline"
                )
                logger.info(f"[RANK {rank}] wandb initialized (offline).")
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
    logger.info(f"Encoder dtype: {next(encoder.parameters()).dtype}")
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

    if eval_in_chans > 1 and attn_pooler_flag is True:
        attn_pooler = AttentionPooling(embed_dim=encoder.model.embed_dim).to(device)  # for multi-channel inputs
    else:
        attn_pooler = None  # Explicitly None for single contrast and multicontrast by default
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
        collator=None,
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
        threshold_isotropy=threshold_isotropy,
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
        collator=None,
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
        threshold_isotropy=threshold_isotropy,
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
        lr_schedule_factor=lr_schedule_factor,
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
        classifier, optimizer, scaler, start_epoch, attn_pooler = load_checkpoint(
            device=device,
            r_path=cls_checkpoint_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler,
            attn_pooler=attn_pooler if (eval_in_chans > 1 and attn_pooler_flag is True) else None
        )
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, train_acc, val_acc, val_f1, auc_score, path, info_path, attn_pooler=None):
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
        if attn_pooler is not None:
            save_dict['attn_pooler'] = attn_pooler.state_dict()
        if rank == 0:
            torch.save(save_dict, path) #rather than to save on latest_path pretrained classifier
            with open(info_path, "w") as info_f:
                info_f.write(
                    f"Model path: {path},\n"
                    f"Epoch: {epoch + 1},\n"
                    f"train acc: {train_acc}, val acc: {val_acc},\n"
                    f"val f1: {val_f1}, auc: {auc_score}, lr: {lr}"
                )
            logger.info(f"Checkpoint successfully saved at epoch {epoch} to {path}")
                
    epoch_accs = []
    epoch_val_accs = []
    encoder_frozen = True
    best_val_acc = float('-inf')
    best_val_f1 = float('-inf')
    best_val_auroc = float('-inf')
    val_auroc = float('-inf')

    if pretrained_path == None:
        encoder_warmup = 0

    # if rank == 0:
    #     sanity_check(encoder, classifier, val_loader, device, num_classes)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            logger.info('Starting Epoch %d' % (epoch))

        if not frozen:
            if epoch >= encoder_warmup:
                encoder_frozen = False

        train_acc, train_loss, train_recall, train_precision, train_f1, auc_score, all_outputs_tensor, all_labels_tensor = run_one_epoch(
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

        val_acc, val_loss, val_recall, val_precision, val_f1, auc_score, val_outputs_tensor, val_labels_tensor = run_one_epoch(
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

        # Compute metrics on validation set
        if val_outputs_tensor is not None:
            val_metrics = eval_metrics(val_outputs_tensor, val_labels_tensor)
            val_auroc = val_metrics['auc']*100 #for percent
            val_acc = val_metrics['accuracy']*100
            val_f1 = val_metrics['f1']*100
            val_precision = val_metrics['precision']*100
            val_recall = val_metrics['recall']*100
        
        if rank == 0:
            logger.info(f"Epoch [{epoch}] Validation Metrics:")
            logger.info(f"AUC: {val_auroc:.4f}")
            logger.info(f"Accuracy: {val_acc:.4f}")
            logger.info(f"Precision: {val_precision:.4f}")
            logger.info(f"Recall: {val_recall:.4f}")
            logger.info(f"F1 Score: {val_f1:.4f}")
            logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        #GU_ DEBUG
        #if rank == 0: #print out for all ranks for debugging
        logger.info('[%3d] rank:[%d] val_acc: %.2f%% val_recall: %.2f%% AUC: %.3f%% val_f1: %.2f%% val_precision: %.2f' % (epoch, rank, val_acc, val_recall, val_auroc, val_f1, val_precision))
        
        if csv_logger is not None:   # CSV logging
                csv_logger.log(epoch, val_acc, val_loss.item(), val_recall, val_precision, val_f1, val_auroc)

        # Wandb logging
        if run != None and rank == 0:
            run.log({
                    'train_epoch/acc': train_acc,
                    'train_epoch/loss': train_loss,
                    'train_epoch/recall': train_recall,
                    'train_epoch/precision': train_precision,
                    'train_epoch/f1': train_f1,
                    'val_epoch/acc': val_acc,
                    'val_epoch/loss': val_loss,
                    'val_epoch/auroc': val_auroc,
                    'val_epoch/recall': val_recall,
                    'val_epoch/precision': val_precision,
                    'val_epoch/f1': val_f1
                })
                
        if log_dir is not None:
            # Always save latest model
            save_checkpoint(epoch, train_acc, val_acc, val_f1, val_auroc,
                        latest_path, latest_info_path, attn_pooler=attn_pooler)

            # Check if this is the best model in terms of:
            if val_auroc > best_val_auroc: #if val_acc > best_val_acc:  #if val_f1 > best_val_f1:
                best_val_auroc = val_auroc  # update the best metric  #best_val_acc = val_acc #best_val_f1 = val_f1
                save_checkpoint(epoch, train_acc, val_acc, val_f1, val_auroc,
                            best_path, best_info_path, attn_pooler=attn_pooler)

            # Save periodic checkpoints
            if epoch % save_ckpt_epoch_freq == 0:
                periodic_path = os.path.join(periodic_model_folder, f'{eval_tag}-periodic-epoch-{epoch}.pth.tar')
                periodic_info_path = os.path.join(periodic_model_folder, f'periodic-info-epoch-{epoch}.txt')
                save_checkpoint(epoch, train_acc, val_acc, val_f1, val_auroc,
                            periodic_path, periodic_info_path, attn_pooler=attn_pooler)

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
    auc_score = 0 #AUC initialize to 0

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

        #clips = data[0]  # [ [clip], [clip], ... ] 
        # labels = data[1]
        #clip_indices = data[2]
        # DEBUG
        # contrast_names_batch = data[3]  # list of lists: [contrast][batch_idx]
        # contrast_names_per_sample = [list(x) for x in zip(*contrast_names_batch)]
        # logger.info(f"Contrast names per sample (batch size = {len(contrast_names_per_sample)}):")
        # for i, contrast_list in enumerate(contrast_names_per_sample):
        #     logger.info(f"  Sample {i}: {contrast_list}")

        new_lr = None
        new_wd = None
        if training:
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
        #with torch.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU: move frames to GPU
            # clips = [   #old style loading, not valid anymore
            #     [dij.to(device, non_blocking=True) for dij in di]  # spatial views
            #     for di in data[0]  # temporal clips
            # ]
            labels = data[1].to(device) # [B]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            contrast_masks = data[3].to(device)  # [B, max_contrasts]
            batch_size = labels.size(0)

            # unwrap the data correctly to a tensor of [B, C, T, H, W]
            full_clip = data[0][0].to(device, non_blocking=True)  # explicitly unwrap list to a tensor: B,C,T,H,W

            #DEBUG
            # print("Old input shape:", clips[0][0].shape)
            # print("New input shape:", full_clip.shape)
            # print("Old requires_grad:", clips[0][0].requires_grad)
            # print("New requires_grad:", full_clip.requires_grad)

            encoder_requires_grad = (not frozen and training)
            encoder_context = torch.enable_grad() if encoder_requires_grad else torch.no_grad()

            with encoder_context:
                if eval_in_chans == 1: #single-contrast pipeline
                    encoder_input = [[full_clip]]  # nesting for encoder: [[B,C=1,T,H,W]] i.e. encoder_input[0][0].shape: B C T H W
                    
                    # logger.info(f"DEBUG: encoder_input nesting: {len(encoder_input)} {len(encoder_input[0])}")
                    # logger.info(f"DEBUG: tensor shape: {encoder_input[0][0].shape}")

                    outputs = encoder(encoder_input, clip_indices)[0]  # [B, N, D]
                else:
                    # Multi-contrast pipeline explicitly handling each contrast batch-wise
                    encoded_contrasts = []
                    for c in range(eval_in_chans):
                        contrast_mask = contrast_masks[:, c]  # [B] mask to show which samples in the batch have contrast c
                        valid_indices = contrast_mask.nonzero(as_tuple=True)[0]

                        if len(valid_indices) == 0:
                            continue  # explicitly skip if no valid contrasts in batch

                        contrast_tensor = full_clip[valid_indices, c:c+1]  # [B_valid, 1, T, H, W] for a given contrast, hence ,1,

                        # explicitly wrap contrast tensor to match expected encoder input [[[B,C,T,H,W]]], ie for ClipAggregation
                        encoder_input = [[contrast_tensor]]  # [[[B_valid,1,T,H,W]]]
                        encoded_output = encoder(encoder_input, clip_indices)[0]  # [B_valid, N, D] valid samples in the batch for contrast c

                        # explicitly create full-batch tensor initialized to zeros to maintain order
                        batch_encoded_contrast = torch.zeros(batch_size, *encoded_output.shape[1:], device=device, dtype=encoded_output.dtype)
                        batch_encoded_contrast[valid_indices] = encoded_output

                        encoded_contrasts.append(batch_encoded_contrast)

                    if len(encoded_contrasts) == 0:
                        raise ValueError("No valid contrasts in entire batch.")

                    # 1. Cancelled this: explicitly aggregate multiple contrasts using attention pooling: [B, N, D]
                    #if attn_pooler is not None: #not needed, this is already checked in initialization
                    # outputs = attn_pooler(encoded_contrasts, contrast_masks) #list of len:B, each with tensors NxD
                    # 2. Concatenate encoded contrasts along token dimension: [B, C*N, D]
                    outputs = torch.cat(encoded_contrasts, dim=1)
                    
            classifier_requires_grad = training  # classifier explicitly trainable during training
            classifier_context = torch.enable_grad() if classifier_requires_grad else torch.no_grad()

            with classifier_context:
                if attend_across_segments:
                    if eval_in_chans == 1: 
                        classifier_outputs = [classifier(outputs)]  # single contrast      
                    else: # masked attention pooling
                        classifier_outputs = [classifier(outputs, contrast_masks)]  # list of B,num_classes tensor: i.e. [B, num_classes]                                
                else: # FIX_ME_Not_needed_for_single_contrast: Does not handle multicontrast case for attend_across_segments is false
                    classifier_outputs = [[classifier(outputs)]]

            # Loss calculation
            if attend_across_segments:
                loss = sum([criterion(o, labels) for o in classifier_outputs]) / len(classifier_outputs)
            else:
                loss = sum(
                    [sum([criterion(ost, labels) for ost in os]) for os in classifier_outputs]
                ) / len(classifier_outputs) / len(classifier_outputs[0])

            with torch.no_grad():
                if attend_across_segments:
                    classifier_outs = sum([F.softmax(o, dim=1) for o in classifier_outputs]) / len(classifier_outputs)
                else:
                    classifier_outs = sum(
                        [sum([F.softmax(ost, dim=1) for ost in os]) for os in classifier_outputs]
                    ) / len(classifier_outputs) / len(classifier_outputs[0])

            # Metric calculations: classifier_outputs: [B, num_classes] at this point
            preds = classifier_outs.max(dim=1).indices

            # Top-1 Accuracy
            top1_acc = 100. * preds.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)

            # Additional Metrics
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # (Specificity is commented out, activate if needed)
            # cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
            # tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Collect results for AUC during validation
            if not training:
                all_outputs.append(classifier_outs.detach().cpu())
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

                ## EE: Why would i clip gradients if im not stepping?
                if epoch > warmup:
                    if (clip_grad_classifier is not None):
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad_classifier)
                    if not frozen and (clip_grad_encoder is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) # GU added 1/19/2025
                scaler.step(optimizer)
                scaler.update()
            else:

                ## EE: No unscaling here?
                loss.backward()
                if epoch > warmup:
                    if (clip_grad_classifier is not None):
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad_classifier)
                    if not frozen and (clip_grad_encoder is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) # GU added 1/19/2025
                optimizer.step()

            #DEBUG
            #logger.info(f"Encoder structure: {encoder.module}")
            # print(encoder.module.model.blocks[0].attn.qkv.weight.requires_grad)
            # try:
            #     grad = encoder.module.model.blocks[-1].attn.qkv.weight.grad
            #     if grad is not None:
            #         logger.info(f"Grad: [{grad.abs().mean().item():.6f}]")
            #     else:
            #         logger.info("Grad: [None]")
            # except AttributeError as e:
            #     logger.warning(f"Could not access gradient: {e}")
            # total_grad_norm = 0.0
            # num_params = 0
            # for name, param in encoder.module.model.named_parameters():
            #     if param.grad is not None:
            #         total_grad_norm += param.grad.data.norm(2).item() ** 2
            #         num_params += 1
            # if num_params > 0:
            #     total_grad_norm = (total_grad_norm) ** 0.5
            #     logger.info(f"Global Grad L2 Norm: [{total_grad_norm:.6f}] across {num_params} parameters")
            # else:
            #     logger.info("No gradients found on any parameters.")


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

       # Wandb logging every iter
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
    # #end of one epoch : AUC and metric Calculations
    all_outputs_tensor = torch.cat(all_outputs, dim=0) if not training else None
    all_labels_tensor = torch.cat(all_labels, dim=0) if not training else None

    # if rank == 0 and not training:
    #     # Concatenate all outputs and labels
    #     all_outputs_tensor = torch.cat(all_outputs, dim=0)  # shape: [total_samples, num_classes]
    #     all_labels_tensor = torch.cat(all_labels, dim=0)    # shape: [total_samples]
    #     # Convert to numpy arrays
    #     all_outputs_np = all_outputs_tensor.numpy()
    #     all_labels_np = all_labels_tensor.numpy()
    
    #     unique_labels = np.unique(all_labels_np)
    #     if len(unique_labels) < 2:
    #         logger.warning(f"Only one class {unique_labels} present in labels. AUC is undefined.")
    #         auc_score = float('nan')
    #     else:
    #         try:           
    #             number_classes = all_outputs_np.shape[1] # num_classes already available
    #             if number_classes == 2: # Binary classification case
    #                 auc_score = roc_auc_score(all_labels_np, all_outputs_np[:, 1])
    #                 # predicted_labels for computing metrics at threshold = 0.5
    #                 predicted_labels = (all_outputs_np[:, 1] >= 0.5).astype(int)
    #                 # Recalculate all metrics based on thresholded predicted labels
    #                 accuracy = accuracy_score(all_labels_np, predicted_labels)
    #                 precision = precision_score(all_labels_np, predicted_labels)
    #                 recall = recall_score(all_labels_np, predicted_labels)
    #                 f1 = f1_score(all_labels_np, predicted_labels)
    #                 conf_matrix = confusion_matrix(all_labels_np, predicted_labels)

    #             else:
    #                 # Multi-class classification (one-vs-rest)
    #                 auc_score = roc_auc_score(
    #                     all_labels_np,
    #                     all_outputs_np,
    #                     multi_class='ovr',
    #                     average='macro'
    #                 )
    #         except ValueError as e:
    #             logger.warning(f"Could not compute AUC: {e}")
    #             auc_score = float('nan')
    #         # Log and print AUC
    #     logger.info(f"AUC Score (validation) inside run_one_epoch: {auc_score:.4f}")

    torch.cuda.empty_cache()
    return top1_meter.avg, loss, recall_meter.avg, precision_meter.avg, f1_meter.avg, auc_score, all_outputs_tensor, all_labels_tensor

def eval_metrics(outputs_tensor, labels_tensor):
    outputs_np = outputs_tensor.cpu().numpy()
    labels_np = labels_tensor.cpu().numpy()

    metrics = {}

    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class {unique_labels} present in labels. AUC and other metrics may be undefined.")
        metrics.update({
            'auc': float('nan'),
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'confusion_matrix': np.array([[len(labels_np)]])
        })
        return metrics

    num_classes = outputs_np.shape[1]

    if num_classes == 2:
        predicted_probs = outputs_np[:, 1]
        metrics['auc'] = roc_auc_score(labels_np, predicted_probs)

        predicted_labels = (predicted_probs >= 0.5).astype(int)

        metrics.update({
            'accuracy': accuracy_score(labels_np, predicted_labels),
            'precision': precision_score(labels_np, predicted_labels, zero_division=0),
            'recall': recall_score(labels_np, predicted_labels, zero_division=0),
            'f1': f1_score(labels_np, predicted_labels, zero_division=0),
            'confusion_matrix': confusion_matrix(labels_np, predicted_labels)
        })
    else:
        predicted_labels = np.argmax(outputs_np, axis=1)
        metrics['auc'] = roc_auc_score(labels_np, outputs_np, multi_class='ovr', average='macro')

        metrics.update({
            'accuracy': accuracy_score(labels_np, predicted_labels),
            'precision': precision_score(labels_np, predicted_labels, average='macro', zero_division=0),
            'recall': recall_score(labels_np, predicted_labels, average='macro', zero_division=0),
            'f1': f1_score(labels_np, predicted_labels, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(labels_np, predicted_labels)
        })

    return metrics

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
    outputs = encoder(clips, clip_indices) #why does the encoder need clip_indices
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
    scaler,
    attn_pooler=None  # Optional attn_pooler for multi-contrast evaluation, None for single-contrast case
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # Load classifier
        classifier_dict = checkpoint.get('classifier', {})
        msg_cls = classifier.load_state_dict(classifier_dict, strict=True)
        logger.info(f'Loaded pretrained classifier from epoch {epoch} with message: {msg_cls}')

        # Load attn_pooler if it exists and is provided
        if attn_pooler is not None and 'attn_pooler' in checkpoint:
            attn_msg = attn_pooler.load_state_dict(checkpoint['attn_pooler'])
            logger.info(f'loaded attn_pooler with msg: {attn_msg}')
        else:
            attn_pooler = None  # Explicitly set to None if single-contrast or no attn_pooler saved
            logger.info('No attn_pooler loaded (single-contrast eval or missing from checkpoint)')

        # Load optimizer and scaler
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'Loaded optimizer and scaler from epoch {epoch}')
        logger.info(f'Checkpoint read from: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        epoch = 0

    # Return updated models and optimizer states explicitly
    return classifier, opt, scaler, epoch, attn_pooler



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
    collator=None,
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
    threshold_isotropy=1.4,
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
        collator=collator,
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
        threshold_isotropy=threshold_isotropy,
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
    lr_schedule_factor,
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
        T_max=int(lr_schedule_factor*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(lr_schedule_factor*num_epochs*iterations_per_epoch))
    
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None 
    
    # **Debugging: Print parameter names, shapes, and requires_grad**
    # print("\nOptimizer Parameter Groups Debug Info:")
    # for group_idx, group in enumerate(optimizer.param_groups):
    #     print(f"\n--- Parameter Group {group_idx + 1} ---")
    #     for name, param in zip(group.get('names', []), group['params']):
    #         print(f"Name: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")

    return optimizer, scaler, scheduler, wd_scheduler
