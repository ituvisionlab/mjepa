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

import copy
import time
import numpy as np

import matplotlib.pyplot as plt #GU_

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.utils.tensorboard
import wandb
import subprocess
import inspect

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
    visualize_fft_3d_spectrum,
)
from app.vjepa.transforms import make_transforms


# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
periodic_ckpt_save_freq = 25 
# --
   
global_step = 0 # global counter for spectral warmup

_GLOBAL_SEED = 0
#np.random.seed(_GLOBAL_SEED)
#torch.manual_seed(_GLOBAL_SEED)
#torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

def main(args, resume_preempt=False, log_dir="./logs/evals", run=None):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    discard_stem = cfgs_meta.get('discard_stem', False)
    reset_schedules = cfgs_meta.get('reset_schedules', False)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    run_ID =  cfgs_meta.get('run_ID', None)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_model_name = cfgs_model.get('pred_model_name','vit_predictor')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    in_chans = cfgs_data.get('in_channel_size', 3)
    random_clip_sampling = cfgs_data.get('random_clip_sampling', False)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [1, 1])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.9, 1.0])
    rot_degree = cfgs_data_aug.get('rotation_degree', 0.0)
    random_noise = cfgs_data_aug.get('random_noise', 0.025)
    random_bias = cfgs_data_aug.get('random_bias', 0.2)
    intensity_gamma = cfgs_data_aug.get('intensity_gamma', 0.2)
    motion_shift = cfgs_data_aug.get('motion_shift', False) #unused
    reprob = cfgs_data_aug.get('reprob', 0.) # unused
    use_aa = cfgs_data_aug.get('auto_augment', False)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff', 0.0)
    spectral_coeff = cfgs_loss.get('spectral_coeff',0.01)
    alpha_vcr = cfgs_loss.get('alpha_vcr',1.0)
    beta_vcr = cfgs_loss.get('beta_vcr',0.1) #0.4

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.95)) #(0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8) #1e-7
    drop_rate = cfgs_opt.get('drop_rate', 0.1)
    attn_drop_rate = cfgs_opt.get('attn_drop_rate', 0.1) 
    accumulation_steps = cfgs_opt.get('grad_accum_steps', 2)  # Define the number of steps before updating the optimizer 

    # -- LOGGING
    cfgs_logging = args.get('logging')
    # folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')
    
    # jepa_ckpt_folder = "/gpfs/data/sodicksonlab/gozde/pretrained_weights"
    jepa_ckpt_folder = cfgs_meta.get("ckpt_folder", "src/models/pretrained_weights")
    
    if log_dir != None:
        model_folder = os.path.join(log_dir, "model_ckpt")
        csv_folder = os.path.join(log_dir, "csv_logs")
        tb_folder = os.path.join(log_dir, "tensorboard")
        
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(tb_folder, exist_ok=True)
        
        
        # Model checkpoint folders
        
        latest_model_folder = os.path.join(model_folder, "latest-model")
        best_model_folder = os.path.join(model_folder, "best-model")
        periodic_model_folder = os.path.join(model_folder, "periodic-model")
        
        os.makedirs(latest_model_folder, exist_ok=True)
        os.makedirs(best_model_folder, exist_ok=True)
        os.makedirs(periodic_model_folder, exist_ok=True)
        
        
        latest_path = os.path.join(latest_model_folder, f'{tag}-latest.pth.tar')
        latest_info_path = os.path.join(latest_model_folder, f'latest-info.txt')
        
        best_path = os.path.join(best_model_folder, f'{tag}-best.pth.tar')
        best_info_path = os.path.join(best_model_folder, f'best-info.txt')
        
    else:
        model_folder = None
        csv_folder = None
        tb_folder = None
        latest_model_folder = None
        best_model_folder = None
        periodic_model_folder = None
        latest_path = None
        latest_info_path = None
        best_path = None
        best_info_path = None
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Ensures seed consistency across GPUs

    # Use deterministic mode if full reproducibility is required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:0')
        torch.cuda.set_device(device)

    # -- load pretrained model path
    load_path = None
    if load_model:
        load_path = os.path.join(jepa_ckpt_folder, r_file) if r_file is not None else None
        if not os.path.exists(load_path):
            load_path = None
            load_model = False
    
    if log_dir != None:
        
        if rank == 0:
            if (run_ID == None):
                # wandb init
                run = wandb.init(
                # set the wandb project where this run will be logged
                project="mjepa-project",
                
                entity="ituvisionlab",
                
                dir=log_dir,

                # track hyperparameters and run metadata
                config=args,
                
                name=os.path.basename(log_dir)
                
                # group="mjepa-DDP"
                )
            else:
                run = wandb.init(
                # set the wandb project where this run will be logged
                project="mjepa-project",
                entity="ituvisionlab",               
                id = run_ID,
                resume="allow",
                dir=log_dir,
                # track hyperparameters and run metadata
                config=args,
                name=os.path.basename(log_dir)
                # group="mjepa-DDP"
                )
        else:
            run = None
        
        # Tensorboard logging
        #tb_rank_folder = os.path.join(tb_folder, f"rank_{rank}")
        #os.makedirs(tb_rank_folder, exist_ok=True)
        log_writer = None # torch.utils.tensorboard.SummaryWriter(tb_rank_folder)
        
        # -- make csv_logger
        log_file = os.path.join(csv_folder, f'{tag}_r{rank}.csv')
        csv_logger = CSVLogger(
            log_file,
            ('%d', 'epoch'),
            ('%d', 'itr'),
            ('%.5f', 'loss'),
            ('%.5f', 'loss-jepa'),
            ('%.5f', 'loss-spec'),
            ('%.5f', 'reg-loss'),
            ('%.5f', 'enc-grad-norm'),
            ('%.5f', 'pred-grad-norm'),
            ('%d', 'gpu-time(ms)'),
            ('%d', 'wall-time(ms)'),
        )
    else:
        tb_rank_folder = None
        log_writer = None
        
        log_file = None
        csv_logger = None
    
    
    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        pred_model_name=pred_model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        in_chans=in_chans,
        use_sdpa=use_sdpa,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        rot_degree = rot_degree,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=True,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         in_chans=in_chans,
         crop_size=crop_size,
         random_clip_sampling=random_clip_sampling,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=csv_folder if log_resource_util_data else None)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataset length: {ipe}/{_dlen}')

    logger.info(f'Dataset len: {len(unsupervised_loader.dataset)}, Num of batches: {_dlen}')
    
    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)
    
    # print("Encoder class:", encoder.__class__)
    # print(inspect.signature(encoder.forward))

    encoder = DistributedDataParallel(encoder, static_graph=True, gradient_as_bucket_view=True)
    predictor = DistributedDataParallel(predictor, static_graph=True, gradient_as_bucket_view=True)
    target_encoder = DistributedDataParallel(target_encoder, gradient_as_bucket_view=True)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch=0
    # -- load training checkpoint
    # if load_model or os.path.exists(load_path):
    if load_model and os.path.exists(load_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            discard_stem=discard_stem)
        
        if reset_schedules:
            start_epoch = 0
        
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            # mask_collator.step() #GU_Debug: not needed anymore

    def save_checkpoint(epoch, path, info_path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
            with open(info_path, "w") as info_f:
                info_f.write(f"Model path: {path},\nEpoch: {epoch}, loss: {loss_meter.avg}, lr: {lr}")
            
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    epoch_losses = []
    warmup_iters = 500 #for spectral loss

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            logger.info('Epoch %d' % (epoch))

        optimizer.zero_grad()

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            iter_start_time = time.time()
           
            #data_start_time = time.time() # **Measure Data Loading Time**
            try:
                udata, masks_enc, masks_pred = next(loader) #returned from "call" of multiblock3d
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                torch.cuda.empty_cache()
                
                loader = iter(unsupervised_loader) #resets the loader iterator again
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            #data_end_time = time.time() # **Measure Data Loading Time End**
            #data_loading_time = data_end_time - data_start_time # **Measure Data Loading Time **
            #logger.info(f"Data Loading Time: {data_loading_time:.4f} sec") # **Measure Data Loading Time End**
             
            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)
            
            #clip_start_time = time.time() # **Measure Load Clips & Transfer to GPU Time**
            clips, masks_enc, masks_pred = load_clips()
            #clip_end_time = time.time() # **Measure Load Clips & Transfer to GPU Time**
            #clip_transfer_time = clip_end_time - clip_start_time # **Measure Load Clips & Transfer to GPU Time**
            #logger.info(f"Clip Transfer Time: {clip_transfer_time:.4f} sec") # **Measure Load Clips & Transfer to GPU Time**

            # if torch.isnan(clips).any():
            #     print("NaN detected in input data!")
            #     raise ValueError("NaN detected in input data!")
            
        # -------------------------------------------------
            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            # cutoff_ratio = 0.3 keeps top 70% of high frequencies, masks bottom 30%
            def make_highpass_mask(shape, cutoff_ratio=0.5, device='cpu'):
                """
                Create a high-pass mask for FFT volumes.
                `cutoff_ratio`: e.g., 0.5 retains upper 50% of frequencies (along each axis)
                """
                B, D, X, Y, Z = shape
                fx = torch.fft.fftfreq(X, d=1).to(device)
                fy = torch.fft.fftfreq(Y, d=1).to(device)
                fz = torch.fft.fftfreq(Z, d=1).to(device)

                # Create meshgrid of frequencies
                grid_fx, grid_fy, grid_fz = torch.meshgrid(fx, fy, fz, indexing="ij")
                freq_magnitude = torch.sqrt(grid_fx ** 2 + grid_fy ** 2 + grid_fz ** 2)

                # Normalize and mask
                freq_norm = freq_magnitude / freq_magnitude.max()
                mask = (freq_norm >= cutoff_ratio).float()
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,X,Y,Z] to broadcast over B and D
                return mask.to(device)

            def spectral_loss_fn_3d(z_early, h_early, mode='complex', debug_visualize=False):
                """
                3D FFT spectral loss between predicted and target early embeddings.
                - z_early, h_early: lists of tensors, each [B, N, D] where N = grid_size_x * grid_size_x * grid_size_d
                - mode: 'complex' (default), or 'magnitude', or 'logmag'
                - Includes NaN/Inf protection
                """
                loss = 0.
                skipped = 0

                grid_size_x = int(crop_size / patch_size)
                grid_size_d = int(num_frames / tubelet_size)
                grid_size = grid_size_d * grid_size_x ** 2
                grid_shape = (grid_size_x, grid_size_x, grid_size_d)

                def flatten_nested_list(lst):
                    return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

                z_early = flatten_nested_list(z_early)
                h_early = flatten_nested_list(h_early)

                for i, (zi, hi) in enumerate(zip(z_early, h_early)):
                    B, N, D = zi.shape
                    assert N == grid_size, f"N={N} is not consistent with grid_size={grid_size}"

                    # Reshape to [B, D, X, Y, Z]
                    X, Y, Z = grid_shape
                    zi = zi.permute(0, 2, 1).contiguous().view(B, D, X, Y, Z)
                    hi = hi.permute(0, 2, 1).contiguous().view(B, D, X, Y, Z)

                    # Compute 3D FFT
                    zi_fft = torch.fft.fftn(zi, dim=(-3, -2, -1))
                    hi_fft = torch.fft.fftn(hi, dim=(-3, -2, -1))

                    # Apply high-pass filter
                    f_mask = make_highpass_mask(zi_fft.shape, cutoff_ratio=0.5, device=zi.device)
                    zi_fft = zi_fft * f_mask
                    hi_fft = hi_fft * f_mask

                    try:
                        if mode == 'complex':
                            loss_r = F.mse_loss(zi_fft.real, hi_fft.real)
                            loss_i = F.mse_loss(zi_fft.imag, hi_fft.imag)
                            if torch.isfinite(loss_r) and torch.isfinite(loss_i):
                                loss += loss_r + loss_i
                            else:
                                skipped += 1

                        elif mode == 'magnitude':
                            mag_loss = F.mse_loss(torch.abs(zi_fft), torch.abs(hi_fft))
                            if torch.isfinite(mag_loss):
                                loss += mag_loss
                            else:
                                skipped += 1

                        elif mode == 'logmag':
                            zi_log = torch.log1p(torch.abs(zi_fft) + 1e-6)
                            hi_log = torch.log1p(torch.abs(hi_fft) + 1e-6)
                            log_loss = F.mse_loss(zi_log, hi_log)
                            if torch.isfinite(log_loss):
                                loss += log_loss
                            else:
                                skipped += 1

                        else:
                            raise ValueError("Mode must be one of: 'complex', 'magnitude', 'logmag'")

                    except Exception as e:
                        print(f"[Warning] Spectral loss computation failed at step {i}: {str(e)}")
                        skipped += 1

                    # Optional FFT visualization
                    if debug_visualize and i == 0:
                        feat = z_early[i]  # [B, N, D]
                        visualize_fft_3d_spectrum(
                            feat, grid_shape, channel_idx=0, slice_dim=2,
                            title_prefix=f"Early Layer {i} FFT Spectrum"
                        )

                if skipped > 0 and len(z_early) > skipped:
                    print(f"[Warning] Skipped {skipped}/{len(z_early)} spectral loss terms due to NaN/Inf")
                elif skipped == len(z_early):
                    print("[Warning] All spectral loss terms skipped — returning zero")
                    return torch.tensor(0.0, device=z_early[0].device, dtype=z_early[0].dtype)

                return loss / (len(z_early) - skipped)


            def train_step():
                global global_step
                spectral_coeff_eff = spectral_coeff * min(1.0, global_step / warmup_iters)
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each mask-pred.
                    """
                    with torch.no_grad():
                        out = target_encoder(c, return_early=True)
                        # print("out keys:", out.keys())  # should show 'early', 'final'

                        h_final = apply_masks(out['final'], masks_pred, concat=False)
                        h_early_full = out['early']  # Keep full early features for FFT
                        return h_final, h_early_full
                        # h = target_encoder(c) #already returns normalized!
                        # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]: This is !!! double normalization!
                        # -- create targets (masked regions of h)
                        #h = apply_masks(h, masks_pred, concat=False)
                        #return h

                def forward_context(c, h_final):
                    """
                    Returns list of tensors of shape [B, N, D], one for each mask-pred.
                    """
                    out = encoder(c, masks_enc, return_early=True)
                    z_input = [o['final'] for o in out]  # list[Tensor], as expected
                    z_early_full = [o['early'] for o in out]
                    h_final = [h for h in h_final]  # convert to list of [B, N, D]

                    if not isinstance(h_final, list): # avoid double-wrapping if h_final is already a list
                        h_final = [h_final]
                    z_final = predictor(z_input, h_final, masks_enc, masks_pred)  # works as before
                    return z_final, z_early_full
                    # z = encoder(c, masks_enc)
                    # z = predictor(z, h, masks_enc, masks_pred)
                    # return z

                def loss_fn(z_final, h_final, z_early, h_early, mode='complex'): # or mode='magnitude' 
                    # JEPA prediction loss
                    loss_pred = 0.
                    for zi, hi in zip(z_final, h_final):
                        loss_pred += torch.mean(torch.abs(zi - hi) ** loss_exp) / loss_exp
                    loss_pred /= len(h_final)

                    # Spectral loss on early layers if desired
                    loss_spec = 0.0
                    if spectral_coeff > 0:
                        loss_spec = spectral_loss_fn_3d(z_early, h_early, mode, debug_visualize=False)  
                    return loss_pred, loss_spec

                # def loss_fn(z, h):
                #     loss = 0.
                #     # Compute loss and accumulate for each mask-enc/mask-pred pair
                #     for zi, hi in zip(z, h):
                #         loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                #     loss /= len(masks_pred)
                #     return loss

                def reg_var_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                def reg_cov_fn(z):
                    """
                    Computes VCR covariance loss across z: list of [B, N, D] tensors
                    Encourages off-diagonal covariance to be small
                    """
                    loss = 0.0
                    for zi in z:
                        # zi: [B, N, D]
                        B, N, D = zi.shape
                        zi = zi.view(-1, D)  # [B*N, D]
                        zi = zi - zi.mean(dim=0, keepdim=True)  # center over batch
                        cov = (zi.T @ zi) / (zi.shape[0] - 1)  # [D, D] covariance matrix
                        off_diag = cov - torch.diag(torch.diag(cov))  # remove diagonal
                        loss += (off_diag ** 2).sum() / D  # normalize by dimension
                    return loss / len(z)

                # Step 1. Forward
                loss_jepa, loss_spec, loss_reg = 0.0, 0.0, 0.0
           
                #forward_start_time = time.time() # **Measure Forward Pass Time**
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # h = forward_target(clips)
                    # z = forward_context(clips, h)
                    # loss_jepa = loss_fn(z, h)  # jepa prediction loss
                    # pstd_z = reg_fn(z)  # predictor variance across patches
                    # loss_reg += torch.mean(F.relu(1.-pstd_z))
                    h_final, h_early = forward_target(clips)
                    z_final, z_early = forward_context(clips, h_final)

                    loss_jepa, loss_spec = loss_fn(z_final, h_final, z_early, h_early)
                    pstd_z = reg_var_fn(z_final)
                    loss_cov = reg_cov_fn(z_final)
                    loss_vcr = alpha_vcr * torch.mean(F.relu(1. - pstd_z)) + beta_vcr * loss_cov
                    loss_reg += loss_vcr

                # Accumulate loss before stepping optimizer
                #loss = (loss_jepa + reg_coeff * loss_reg) / accumulation_steps  # Normalize loss

                # Weighting loss terms
                loss = ((loss_jepa + spectral_coeff_eff * loss_spec + reg_coeff * loss_reg)
                    / accumulation_steps
                )
                
                # forward_end_time = time.time() # **Measure Forward Pass Time**
                # forward_time = forward_end_time - forward_start_time # **Measure Forward Pass Time**
                # logger.info(f"Forward Pass Time: {forward_time:.4f} sec") # **Measure Forward Pass Time**

                
                # backward_start_time = time.time() # **Measure Backward Pass + Optimizer Step**
                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0. 
                torch.cuda.synchronize()
                loss = AllReduce.apply(loss)  # Average loss across GPUs  
                if mixed_precision:
                    scaler.scale(loss).backward()
                    if (itr + 1) % accumulation_steps == 0:  # Only unscale when we're going to step
                        scaler.unscale_(optimizer)
                        if (epoch > warmup) and (clip_grad is not None):
                            _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                            _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                        #torch.cuda.synchronize()
                        #loss = AllReduce.apply(loss)  # Average loss across GPUs  
                else:
                    loss.backward()
                    if (itr + 1) % accumulation_steps == 0:  # Only when we're going to step
                        if (epoch > warmup) and (clip_grad is not None):
                            _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                            _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                        optimizer.step()
                        #torch.cuda.synchronize()
                        #loss = AllReduce.apply(loss)  # Average loss across GPUs
                
                # backward_end_time = time.time() # **Measure Backward Pass + Optimizer Step**
                # backward_time = backward_end_time - backward_start_time # **Measure Backward Pass + Optimizer Step**
                # logger.info(f"Backward Pass Time: {backward_time:.4f} sec") # **Measure Backward Pass + Optimizer Step**

                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optim_stats = adamw_logger(optimizer)

                if (itr + 1) % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # Efficient way to clear gradients
                #optimizer.zero_grad()  # Only zero gradients after step

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss) * accumulation_steps,  # Restore original loss scale
                    float(loss_jepa),
                    float(loss_spec),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_spec, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - iter_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)
            
            global global_step
            global_step += 1 #for warmup of spectral loss coef

            gpu_memory_alloc = torch.cuda.max_memory_allocated() / 1024.0 ** 2 # **Monitor Memory & GPU Utilization**
            # logger.info(f"GPU Memory Allocated: {gpu_memory_alloc:.2f} MB") # **Monitor Memory & GPU Utilization**

            
            # iter_end_time = time.time() # **Total Iteration Time**
            # iter_elapsed_time = iter_end_time - iter_start_time # **Total Iteration Time**
            # logger.info(f"Iteration Time: {iter_elapsed_time:.4f} sec") # **Total Iteration Time**

            # Release memory
            del clips
            #torch.cuda.empty_cache() # do not do this after each iteration, but after each epoch!

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch,
                    itr,
                    loss,
                    loss_jepa,
                    loss_spec,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                
                # Tensorboard logging
                # log_writer.add_scalar('train/loss', loss, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/loss_jepa', loss_jepa, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/loss_reg', loss_reg, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/global_norm', grad_stats.global_norm, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/pred_global_norm', grad_stats_pred.global_norm, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/gpu_etime_ms', gpu_etime_ms, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/iter_elapsed_time_ms', iter_elapsed_time_ms, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/memory', gpu_memory_alloc, (epoch * ipe) + itr)
                # log_writer.flush()
                
                
                # Wandb logging
                if run != None and rank == 0:
                    run.log({
                            'train/loss': loss,
                            'train/loss_jepa': loss_jepa,
                            'train/loss_spec': loss_spec,
                            'train/loss_reg': loss_reg,
                            'train/global_norm': grad_stats.global_norm,
                            'train/pred_global_norm': grad_stats_pred.global_norm,
                            'train/gpu_etime_ms': gpu_etime_ms,
                            'train/iter_elapsed_time_ms': iter_elapsed_time_ms,
                            'train/memory': gpu_memory_alloc,
                            'train/lr': _new_lr,
                            'train/wd': _new_wd
                        })
                
            def info_stats():
                
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg,
                           wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch, itr,
                               optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min,
                               optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min,
                               optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max,
                               grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch, itr,
                               grad_stats_pred.first_layer,
                               grad_stats_pred.last_layer,
                               grad_stats_pred.min,
                               grad_stats_pred.max,
                               grad_stats_pred.global_norm))
            
            if log_dir != None:
                log_stats()
                
            info_stats()
                
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint
        logger.info('--- Epoch avg. loss %.3f ---' % loss_meter.avg)
        
        # DEBUG_ save current ckpt for debugging
        # temp_log_dir = "/gpfs/home/unalg01/jepa/mjepa_ckpt.pth"
        # temp_latest_info_path='/gpfs/home/unalg01/jepa/mjepa_latest-info.txt'
        # save_checkpoint(epoch, temp_log_dir, temp_latest_info_path)

        # -- Save Last
        #if ((itr == 0) and epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1)) and log_dir != None:
        if log_dir != None: # itr is always ipe-1 at this point, do at the end of every epoch   
            if not os.path.exists(latest_path):
                save_checkpoint(epoch, latest_path, latest_info_path)
            else:
                if len(epoch_losses) > 0:
                    if loss_meter.avg < min(epoch_losses) and epoch > 19 :
                        save_checkpoint(epoch, best_path, best_info_path)
                    else:
                        save_checkpoint(epoch, latest_path, latest_info_path)
                    if epoch % periodic_ckpt_save_freq == 0:
                        periodic_path = os.path.join(periodic_model_folder, f'{tag}-periodic-epoch-{epoch}.pth.tar')
                        periodic_info_path = os.path.join(periodic_model_folder, f'periodic-info-epoch-{epoch}.txt')
                        save_checkpoint(epoch, periodic_path, periodic_info_path)
        if epoch >= 19:
            epoch_losses.append(loss_meter.avg)

        # End of one epoch
        torch.cuda.empty_cache() 

        # SUBMIT A Classifier Evaluation Periodically
        #if epoch % 150 == 0:
        #    subprocess.call(['sbatch', './test.sh']) 

    if run != None:
        run.finish()
