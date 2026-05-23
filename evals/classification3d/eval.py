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

# NOTE: 5th March, Esra: Trying to make the code compatible with UHEM.

"""
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass
"""
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

import torch.utils.tensorboard
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

import sys 
sys.path.append('/ari/users/eergun01/jepa')
#sys.path.append('/home/gozde/medChangeDet/jepa')

import src.models.resnet3d as resnet

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
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from evals.classification3d.utils import (
    make_transforms,
    make_video_transforms,
    ClipAggregation,
    FrameAggregation
)
from src.utils.tensors import trunc_normal_

# logging.basicConfig(filename='my_log_file.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
    in_chans = args_pretrain.get('in_channel_size', 3)
    frozen = args_pretrain.get('frozen', True)
    encoder_warmup = args_pretrain.get('encoder_warmup', 1)
    use_pos_embed = args_pretrain.get('use_pos_embed', False)
    clip_grad_encoder = args_pretrain.get('clip_grad_encoder',1.0)

    # -- DATA
    args_data = args_eval.get('data')
    # train_data_path = [args_data.get('dataset_train')]
    train_data_path = args_data.get('dataset_train', [])
    val_data_path = args_data.get('dataset_val', []) #[args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_clips = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
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
    drop_rate = args_opt.get('drop_rate', 0.1)
    attn_drop_rate = args_opt.get('attn_drop_rate', 0.1) 
   
    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None) # tag: k400-16x8x3
    cls_checkpoint_path = args_eval.get('checkpoint_path')

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # NOTE: Esra: Changing the following to make it compatible with UHEM.


    """
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    """

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        local_rank = -1
    else:
        # works on slurm and also torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    
    checkpoint_freq = 1
    
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
                                ('%.5f', 'val loss'))
        
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

    # Initialize model: Resnet3D encoder with a classifier output
    encoder = init_model(
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        num_classes=num_classes,
        in_chans=in_chans,
        checkpoint_key="classifier"
        )
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(encoder).to(device)

    print("Print the classifier")
    print(encoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Classifier number of parameters: {count_parameters(encoder)}')
    
    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_clips=eval_num_clips, #if attend_across_segments else 1,
        num_views_per_segment=1,
        in_chans=in_chans,
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
        in_chans=in_chans,
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

    # NOTE: Esra: I'm commenting out the following to make the code compatible with UHEM.
    """
    encoder = DistributedDataParallel(encoder, static_graph=True, gradient_as_bucket_view=True) #GU_Debug
    """
    if torch.cuda.is_available():
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank)
    else:
        ddp_kwargs = {}
    encoder = DistributedDataParallel(
            encoder, **ddp_kwargs, static_graph=True, gradient_as_bucket_view=True
        )
    # -- load training checkpoint
    start_epoch = 0
 
    def save_checkpoint(epoch, train_acc, val_acc, path, info_path):
        save_dict = {
            'encoder': encoder.state_dict(),  # Save encoder state
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


    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            logger.info('Epoch %d' % (epoch + 1))

        train_acc, train_loss = run_one_epoch(
            device=device,
            training=True,
            num_temporal_views=eval_num_clips, #if attend_across_segments else 1,
            attend_across_segments=attend_across_segments,
            num_spatial_views=1,
            encoder=encoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            data_sampler=train_sampler,
            use_bfloat16=use_bfloat16,
            log_writer=log_writer,
            epoch=epoch,
            eval_freq=train_eval_freq,
            rank=rank,
            run=run,
            num_classes=num_classes,
            warmup=warmup,
            clip_grad_encoder=clip_grad_encoder,
            clip_grad_classifier=clip_grad_classifier)

        val_acc, val_loss = run_one_epoch(
             device=device,
             training=False,
             num_temporal_views=eval_num_clips,
             attend_across_segments=attend_across_segments,
             num_spatial_views=eval_num_views_per_segment,
             encoder=encoder,
             scaler=scaler,
             optimizer=optimizer,
             scheduler=scheduler,
             wd_scheduler=wd_scheduler,
             data_loader=val_loader,
             data_sampler=val_sampler,
             use_bfloat16=use_bfloat16,
             log_writer=log_writer,
             epoch=epoch,
             eval_freq=val_eval_freq,
             rank=rank,
             run=run,
             num_classes=num_classes,
             warmup=warmup,
             clip_grad_encoder=clip_grad_encoder,
             clip_grad_classifier=clip_grad_classifier)

        if rank == 0:
            logger.info('[%5d] train: %.3f%% test: %.3f%%' % (epoch + 1, train_acc, val_acc))
        
        # if rank == 0:
        if csv_logger != None:
            csv_logger.log(epoch + 1, train_acc, val_acc, train_loss, val_loss)
        
        if (epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1)) and log_dir != None:
            
            if not os.path.exists(latest_path):
                save_checkpoint(epoch + 1, train_acc, val_acc, latest_path, latest_info_path)
            else:
                if len(epoch_accs) > 0:
                    if val_acc > max(epoch_accs) and epoch > 4:
                        save_checkpoint(epoch + 1, train_acc, val_acc, best_path, best_info_path)
                    elif epoch%20==0:
                        periodic_path = os.path.join(periodic_model_folder, f'{eval_tag}-periodic-epoch-{epoch+1}.pth.tar')
                        periodic_info_path = os.path.join(periodic_model_folder, f'periodic-info-epoch-{epoch+1}.txt')
                        save_checkpoint(epoch + 1, train_acc, val_acc, periodic_path, periodic_info_path)
                    else:
                        save_checkpoint(epoch + 1, train_acc, val_acc, latest_path, latest_info_path)
        if epoch > 4:
            epoch_accs.append(train_acc)
            epoch_val_accs.append(val_acc)
            
    if run != None:
        run.finish()


def run_one_epoch(
    device,
    training,
    encoder,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    data_sampler,
    use_bfloat16,
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
    clip_grad_classifier
):

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
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
            # reshape clips to B,C,T,W,H
            clips = torch.cat(clips[0], dim=0).unsqueeze(1)   # shape [batch_size, C, T, W, H]
            # print("Input to encoder:", clips.shape)

            # clips list: len = no_of_clips 
            # e.g. clips[0][0].shape -> torch.Size([4, 3, 16, 224, 224]): B x C x T X W X H
            # clips[1][0].shape ""
            # Forward and prediction
            outputs = None
            if training:
                outputs = encoder(clips)
            else:
                with torch.no_grad():
                    outputs = encoder(clips)

        # Compute loss
        loss = criterion(outputs, labels)
        with torch.no_grad():
            outputs_softmax = F.softmax(outputs, dim=1)
            
            top1_acc = 100. * outputs_softmax.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)
            
            # Compute additional metrics per batch
            preds = outputs_softmax.max(dim=1).indices
           # Use macro average for multiclass classification
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro') # average over all classes
            # recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average=None) # per class average
            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
           
           # cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
           # tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
           # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Collect results for AUC during validation
            if not training:
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())
            # auroc calculations
            logits = outputs.max(dim=1).values
            auroc = roc_auc_score(labels.cpu().numpy(), logits.cpu().numpy(), labels=np.arange(num_classes))
            if len(set(labels.cpu().numpy())) > 1:
                auroc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy()[:, 1])
            else:
                auroc = 0  # float('nan') 

            # Reduce metrics across GPUs
            # NOTE: Esra: Correcting the following for new setup. 
            # auroc = float(AllReduce.apply(torch.tensor(auroc, device='cuda')))
            """
            recall = float(AllReduce.apply(torch.tensor(recall, device='cuda')))
            precision = float(AllReduce.apply(torch.tensor(precision, device='cuda')))
            #specificity = float(AllReduce.apply(torch.tensor(specificity, device='cuda')))
            f1 = float(AllReduce.apply(torch.tensor(f1, device='cuda')))
            """
            auroc = float(AllReduce.apply(torch.tensor(auroc, device=device)))
            recall = float(AllReduce.apply(torch.tensor(recall, device=device)))
            precision = float(AllReduce.apply(torch.tensor(precision, device=device)))
            #specificity = float(AllReduce.apply(torch.tensor(specificity, device=device)))
            f1 = float(AllReduce.apply(torch.tensor(f1, device=device)))            
            auroc_meter.update(auroc)
            recall_meter.update(recall)
            precision_meter.update(precision)
            # specificity_meter.update(specificity)
            f1_meter.update(f1)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if epoch > warmup:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) 
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if epoch > warmup:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_encoder) 
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
        
        # Wandb logging
        if run != None and rank == 0:
            if training and itr % eval_freq == 0:
                run.log({
                        'train/acc': top1_meter.avg,
                        'train/loss': loss,
                        'train/auroc': auroc_meter.avg,
                        'train/recall': recall_meter.avg,
                        'train/precision': precision_meter.avg,
                        # 'train/specificity': specificity_meter.avg,
                        'train/f1': f1_meter.avg,
                        'train/mem': torch.cuda.max_memory_allocated() / 1024.**2,
                        'train/lr': new_lr,
                        'train/wd': new_wd
                    })       
        
        if not training: #end of epoch metrics calculations
            all_outputs_tensor = torch.cat(all_outputs, dim=0) if all_outputs else None
            all_labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None

            if all_outputs_tensor is not None:
                val_metrics = eval_metrics(all_outputs_tensor, all_labels_tensor)

            val_acc = val_metrics['accuracy'] * 100
            val_f1 = val_metrics['f1'] * 100
            val_auc = val_metrics['auc'] * 100
            val_precision = val_metrics['precision'] * 100
            val_recall = val_metrics['recall'] * 100
            
        # Wandb logging
        if not training and run != None and itr % eval_freq == 0:
            run.log({
                        'val_epoch/acc': val_acc,
                        'val_epoch/loss': loss,
                        'val_epoch/auroc': val_auc,
                        'val_epoch/recall': val_recall,
                        'val_epoch/precision': val_precision,
                        'val_epoch/f1': val_f1,
                        'val/mem': torch.cuda.max_memory_allocated() / 1024.**2
                    })
            if rank == 0:
                logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                            % (itr, top1_meter.avg, loss,
                            torch.cuda.max_memory_allocated() / 1024.**2))
                logger.info(f"[FINAL EPOCH METRICS] AUC: {val_auc:.2f}, Acc: {val_acc:.2f}, F1: {val_f1:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}")
                logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        torch.cuda.empty_cache()
        
    return top1_meter.avg, loss

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
    num_classes=2,
    in_chans=1,
    checkpoint_key="classifier"
):
    encoder = resnet.__dict__[model_name](
        num_classes=num_classes,
        in_channels=in_chans
    )

    if pretrained is not None:
        encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    encoder.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Encoder number of parameters: {count_parameters(encoder)}')

    return encoder

def init_opt(
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
    param_groups =[]
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
