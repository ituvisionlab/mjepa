# mjepa/mae: A 3D MRI self-supervised learning framework based on a modified V-JEPA
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

import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.utils.tensorboard
import wandb
import subprocess
import nibabel as nib

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

from app.mae.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
    patchify_image,
    unpatchify_image,
    unpatchify_image_from_full
)
from app.mae.transforms import make_transforms


# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
periodic_ckpt_save_freq = 25
write_img_freq = 10 #20 # every other x epochs, save reconstructed images periodically for monitoring
# --

global_step = 0 # global counter for spectral warmup
global_grad_enc = 0.0

_GLOBAL_SEED = 0
#np.random.seed(_GLOBAL_SEED)
#torch.manual_seed(_GLOBAL_SEED)
#torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

def main(args, resume_preempt=False, log_dir="./logs/evals", run=None):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    torch.cuda.empty_cache()
    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
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
    pred_model_name = cfgs_model.get('pred_model_name','vit_decoder')
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
    reg_coeff = cfgs_loss.get('reg_coeff')
    spectral_coeff = cfgs_loss.get('spectral_coeff',0.01)
    spec_loss_every_n_iter = cfgs_loss.get('spec_loss_every_n_iter',25)    
    min_epoch_for_spectral = cfgs_loss.get('min_epoch_for_spectral',10)

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
    betas = cfgs_opt.get('betas', (0.9, 0.95)) #GU_Debug: (0.9, 0.999)
    eps = cfgs_opt.get('eps', 1.e-8) #1e-7
    drop_rate = cfgs_opt.get('drop_rate', 0.1)
    attn_drop_rate = cfgs_opt.get('attn_drop_rate', 0.1) 
    accumulation_steps = cfgs_opt.get('grad_accum_steps', 2)  # Define the number of steps before updating the optimizer 

    # -- LOGGING
    cfgs_logging = args.get('logging')
    # folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')
    
    # mae_ckpt_folder = "/gpfs/data/sodicksonlab/gozde/pretrained_weights"
    mae_ckpt_folder = cfgs_meta.get("ckpt_folder", "src/models/pretrained_weights")
    
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
        #device = torch.device(f'cuda:0')
        device = torch.device('cuda', rank % torch.cuda.device_count())  # safer for multi-GPU
        torch.cuda.set_device(device)

    # -- load pretrained model path
    load_path = None
    if load_model:
        load_path = os.path.join(mae_ckpt_folder, r_file) if r_file is not None else None #latest_path
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
            ('%.5f', 'loss-mae'),
            ('%.5f', 'loss-spec'),
            ('%.5f', 'loss-reg'),
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
    encoder, decoder = init_video_model(
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
        decoder=decoder,
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
    encoder = DistributedDataParallel(encoder, static_graph=True, gradient_as_bucket_view=True)
    decoder = DistributedDataParallel(decoder, static_graph=True, gradient_as_bucket_view=True)

    start_epoch=0
    # -- load training checkpoint
    # if load_model or os.path.exists(load_path):
    if load_model and os.path.exists(load_path):
        (
            encoder,
            decoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            decoder=decoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            # mask_collator.step() #GU_Debug: not needed anymore

    def save_checkpoint(epoch, path, info_path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
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
        mae_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred = next(loader) #returned from "call" of multiblock3d
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                torch.cuda.empty_cache()
                
                loader = iter(unsupervised_loader) #resets the loader iterator again
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            assert len(masks_enc) == len(masks_pred) == 1, "MAE only supports single-level masking"

            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch dimension
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
            clips, masks_enc, masks_pred = load_clips()

        # -------------------------------------------------
            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            # e.g. cutoff_ratio = 0.3 keeps top 70% of high frequencies, masks bottom 30%
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

            def spectral_loss_images(img_recon, img_gt, mode='complex', cutoff_ratio=0.5):

                """
                Computes 3D FFT spectral loss between reconstructed and original volumes.
                img_recon, img_gt: torch tensors of shape [B, T, H, W]
                """
                assert img_recon.shape == img_gt.shape, "Mismatch in image shapes"
                B, T, H, W = img_recon.shape

                # Normalize both recon and gt to zero-mean, unit-std
                def normalize(x):
                    x = x - x.mean(dim=(1, 2, 3), keepdim=True)
                    x_std = x.std(dim=(1, 2, 3), keepdim=True)
                    return x / (x_std + 1e-6)

                img_recon = normalize(img_recon)
                img_gt = normalize(img_gt)

                if rank == 0  and (epoch % write_img_freq == 0):
                    recon_np = img_recon[0].detach().cpu().numpy()
                    target_np = img_gt[0].detach().cpu().numpy()
                    nib.save(nib.Nifti1Image(recon_np, affine=np.eye(4)), f"Zdebug_recon_step{global_step}.nii.gz")
                    nib.save(nib.Nifti1Image(target_np, affine=np.eye(4)), f"Zdebug_target_step{global_step}.nii.gz")


                # Optional: clamp and stabilize inputs before FFT
                # img_recon = torch.clamp(img_recon, -1e6, 1e6)
                # img_gt = torch.clamp(img_gt, -1e6, 1e6)
   
                # FFT: img_recon, img_gt: shape [B, T, H, W]
                fft_recon = torch.fft.fftn(img_recon, dim=(-3, -2, -1))
                fft_target = torch.fft.fftn(img_gt, dim=(-3, -2, -1))

                # Optional: high-pass filtering
                # f_mask = make_highpass_mask(fft_recon.shape, cutoff_ratio=cutoff_ratio, device=recon.device)
                # fft_recon *= f_mask
                # fft_target *= f_mask

                # Optional: Normalize each FFT volume by its norm across spatial dims
                # fft_recon = fft_recon / (torch.linalg.vector_norm(fft_recon, dim=(-3, -2, -1), keepdim=True) + 1e-6)
                # fft_target = fft_target / (torch.linalg.vector_norm(fft_target, dim=(-3, -2, -1), keepdim=True) + 1e-6)

                # Debug: Monitor magnitudes before loss
                with torch.no_grad():
                    if rank == 0  and (epoch % write_img_freq == 0):
                        spec_r = torch.abs(fft_recon[0]).log1p()
                        spec_t = torch.abs(fft_target[0]).log1p()
                        plt.imsave(f"Zspec_recon_xy_{global_step}.png", spec_r[:, :, spec_r.shape[-1]//2].cpu().numpy(), cmap='gray')
                        plt.imsave(f"Zspec_target_xy_{global_step}.png", spec_t[:, :, spec_t.shape[-1]//2].cpu().numpy(), cmap='gray')
                        mag_recon = torch.abs(fft_recon).max().item()
                        mag_target = torch.abs(fft_target).max().item()
                        mean_r = torch.abs(fft_recon).mean().item()
                        mean_t = torch.abs(fft_target).mean().item()
                        if mag_recon > 1e2 or mag_target > 1e2:
                            logger.warning(f"[SpectralLoss Debug] High FFT magnitude! max_r={mag_recon:.2e}, max_t={mag_target:.2e}")
                            logger.info(f"[SpectralLoss Debug] mean_r={mean_r:.2e}, mean_t={mean_t:.2e}")

                # Compute spectral loss
                if mode == 'complex':
                    loss = F.mse_loss(fft_recon.real, fft_target.real) + F.mse_loss(fft_recon.imag, fft_target.imag)
                elif mode == 'magnitude':
                    loss = F.mse_loss(torch.abs(fft_recon), torch.abs(fft_target))
                elif mode == 'logmag':
                    recon_log = torch.log1p(torch.abs(fft_recon) + 1e-6)
                    target_log = torch.log1p(torch.abs(fft_target) + 1e-6)
                    loss = F.mse_loss(recon_log, target_log)
                else:
                    raise ValueError("Mode must be 'complex', 'magnitude', or 'logmag'")

                # Final safeguard, clip huge losses!!!
                if not torch.isfinite(loss) or loss > 1e3:
                    #logger.warning(f"[SpectralLoss] Clipping extreme value: {loss.item():.2e}")
                    loss = torch.tensor(0.0, device=loss.device)

                return loss


            def train_step_dummy():
                # --- Dummy Input ---
                B, C, T, H, W = 2, 1, 16, 128, 128
                dummy_clips = torch.rand(B, C, T, H, W, device=device)

                # --- Forward Encoder ---
                z, masks_enc, masks_pred = encoder(dummy_clips, return_all_tokens=True)

                logger.debug(f"[DEBUG] Dummy clip shape: {dummy_clips.shape}")
                logger.debug(f"[DEBUG] Encoder output shape (z[0]): {z[0].shape}")
                logger.debug(f"[DEBUG] masks_enc[0]: {masks_enc[0].shape}, masks_pred[0]: {masks_pred[0].shape}")

                # --- Decoder ---
                c_hat = decoder(z[0], masks_enc[0], masks_pred[0])

                logger.debug(f"[DEBUG] Decoder output shape: {c_hat.shape}")
                logger.debug(f"[DEBUG] c_hat requires_grad: {c_hat.requires_grad}")

                # --- Target patches ---
                patches = patchify_image(dummy_clips, patch_size)
                target_patches = apply_masks(patches, masks_pred, concat=False)  # returns list
                logger.debug(f"[DEBUG] Target patch shape: {target_patches[0].shape}")

                # --- MAE Loss ---
                loss = 0.
                for pred, tgt in zip(c_hat, target_patches):
                    loss += F.l1_loss(pred, tgt)
                loss /= len(c_hat)

                logger.info(f"[DEBUG] Dummy MAE loss: {loss.item():.4f}")

                # --- Backprop ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --- Grad Check (e.g., decoder layer) ---
                logger.debug(f"[DEBUG] decoder.predictor_proj.weight.grad norm: {decoder.predictor_proj.weight.grad.norm().item():.4f}")

            def train_step():
                global global_step
                global global_grad_enc #DEBUG May30
                if global_step % spec_loss_every_n_iter == 0:
                    spectral_coeff_eff = min(spectral_coeff * min(1.0, global_step / warmup_iters)* spec_loss_every_n_iter, 0.1) #cap to 0.1
                else:
                    spectral_coeff_eff = 0.0
                #spectral_coeff_eff = spectral_coeff * min(1.0, global_step / warmup_iters)
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_context(c):
                    """
                    Returns dummy context tokens for debugging decoder training independently.
                    """
                    B = c.shape[0]
                    N = masks_enc[0].shape[1]  # number of visible tokens
                    D = encoder.module.backbone.embed_dim  # or pred_embed_dim if needed

                    # Create dummy encoder output (random input to test decoder learning)
                    z = torch.randn(B, N, D, device=c.device, requires_grad=True)

                    # Forward through decoder with dummy tokens
                    c_pred = decoder(z, masks_enc[0], masks_pred[0])  # shape: [B, N_masked, D_out]

                    logger.info(f"[DEBUG] Dummy z shape: {z.shape}")
                    logger.info(f"[DEBUG] c_pred shape: {c_pred.shape}")
                    logger.info(f"[DEBUG] decoder output mean: {c_pred.mean().item():.4f}, std: {c_pred.std().item():.4f}")
                    logger.info(f"[DEBUG] z.requires_grad: {z.requires_grad}, c_pred.requires_grad: {c_pred.requires_grad}")
                    logger.info(f"[DEBUG] c_pred.requires_grad: {c_pred.requires_grad}")

                    return c_pred, [z]  # mimic original return structure

                # def forward_context(c):
                #     """
                #     Returns list of tensors of shape [B, N, D], one for each mask-pred.
                #     """
                #     z = encoder(c, masks_enc)
                #     c_pred = decoder(z[0], masks_enc[0],  masks_pred[0]) #decoder is not multimask wrapped, so strip out of list
                    
                #     logger.info(f"[DEBUG] z[0] shape: {z[0].shape}") #Debug May30
                #     logger.info(f"[DEBUG] c_pred shape: {c_pred[0].shape}") #Debug May30
                #     logger.info(f"[DEBUG] decoder output mean: {c_pred[0].mean().item():.4f}, std: {c_pred[0].std().item():.4f}")
                #     logger.info(f"[DEBUG] Num masked tokens: {masks_pred[0].shape}")
                #     logger.info(f"[DEBUG] c_pred requires_grad: {c_pred.requires_grad}")
                #     logger.info(f"[DEBUG] z requires_grad: {z[0].requires_grad}")

                #     return c_pred, z

                def forward_context_full(c, z):
                    """
                    Returns list of tensors of shape [B, N, D], one for each mask-pred.
                    """
                    ctxt_tokens, tgt_tokens = decoder(z[0], masks_enc[0],  masks_pred[0], return_all_tokens=True)
                    
                    return ctxt_tokens, tgt_tokens
                
                def loss_fn(c_hat, c):
                    patches = patchify_image(c, patch_size)

                    # get only target patches
                    target_patches = apply_masks(patches, masks_pred, concat=False) #returns a list w/concat false
                    # Check apply_masks Alignment
                    logger.info(f"[DEBUG] target shape: {target_patches[0].shape}") #Debug May30
                    logger.info(f"[DEBUG] c_hat requires_grad: {c_hat.requires_grad}")
                    # make sure loss gradients can flow back through decoder → encoder:
                    c_hat[0, 0, 0].backward(retain_graph=True) # REMOVE THIS AFTER DEBUG!!!
                    if encoder.module.backbone.patch_embed.proj.weight.grad is not None:
                        logger.info(f"[DEBUG] c_hat Grad norm: {encoder.module.backbone.patch_embed.proj.weight.grad.abs().sum().item():.4e}")
                    else:
                        logger.warning("[DEBUG] No grad: encoder.module.backbone.patch_embed.proj.weight")
                    logger.info(f"[DEBUG] Num masked target tokens: {masks_pred[0].numel()}")
                    logger.info(f"[DEBUG] Target patch std: {target_patches[0].std().item():.4f}")
                    logger.info(f"[DEBUG] Pred patch std: {c_hat.std().item():.4f}")
                    logger.info(f"[DEBUG] mask_enc size: {masks_enc[0].shape}, mask_pred size: {masks_pred[0].shape}")


                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for pi, ti in zip(c_hat, target_patches):
                        loss += torch.mean(torch.abs(pi - ti)**loss_exp) / loss_exp
                    loss /= len(masks_pred)                   
                    return loss

                def reg_var_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
               
                def reconstruct_image(z, c):
                    # c: original video/image, z: reconstructed patches for each mask level
                    if not isinstance(z, list):
                        z = [z]
                    patches = patchify_image(c, patch_size)
                    # get only unmasked patches from the image for each mask level.
                    nonmasked_patches = apply_masks(patches, masks_enc, concat=False)  # returns a list
                    
                    imgs = []
                    # For each mask level, pass the corresponding mask indices.
                    for level, (recon_tokens, unmask_tokens) in enumerate(zip(z, nonmasked_patches)):
                        imgs.append(
                            unpatchify_image(
                                recon_tokens,
                                unmask_tokens,
                                patch_size,
                                tubelet_size,
                                num_frames,
                                in_chans,
                                crop_size,
                                masks_enc[level],  # use mask for the current level
                                masks_pred[level]  # use mask for the current level
                            )
                        )
                    return imgs
                
                def reconstruct_image_full(z_full):
                    """
                    z_full: list of all tokens reconstructed for each level, each of shape [B, N, D]
                    Assumes decoder was given full embeddings and predicts full set of patches.
                    """
                    imgs = []
                    for recon_tokens in z_full:
                        img = unpatchify_image_from_full(
                            recon_tokens,
                            patch_size,
                            tubelet_size,
                            num_frames,
                            in_chans,
                            crop_size,
                        )
                        imgs.append(img)
                    return imgs

                def reconstruct_mask_volume(masks_pred, patch_size, tubelet_size, num_frames, crop_size):
                    """
                    Create a binary 3D volume for each mask level, showing the locations of the masked patches.
                    
                    Args:
                        masks_pred (list of torch.Tensor): List of tensors, one per mask level.
                            Each tensor should have shape (B, L_masked). We use the first sample in the batch.
                        patch_size (int): Spatial patch size.
                        tubelet_size (int): Temporal patch (tubelet) size.
                        num_frames (int): Total number of frames in the video.
                        crop_size (int): Spatial crop size (assumed square).
                    
                    Returns:
                        volumes (list of torch.Tensor): A list of 3D binary volumes (one per mask level),
                            each of shape (num_frames, crop_size, crop_size). Masked patches are marked with 1.
                    """
                    # Compute grid sizes.
                    grid_spatial = crop_size // patch_size   # number of patches per spatial dimension
                    grid_temporal = num_frames // tubelet_size  # number of patch blocks along temporal dimension
                    
                    volumes = []
                    
                    # For each mask level...
                    for level in range(len(masks_pred)):
                        # Select the masked patch indices for the first sample in the batch.
                        # They should be indices in the range [0, grid_temporal * grid_spatial * grid_spatial).
                        mask_pred_level = masks_pred[level][0]  # shape: (L_masked,)
                        
                        # Create an empty volume: (num_frames, crop_size, crop_size)
                        vol = torch.zeros(num_frames, crop_size, crop_size, dtype=torch.uint8)
                        
                        # For each patch index, compute its grid coordinates and set the corresponding block to 1.
                        for idx in mask_pred_level:
                            # Make sure idx is an integer
                            idx = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
                            
                            # Determine which patch block (in the grid) this index corresponds to.
                            # Total patches per temporal slice is grid_spatial * grid_spatial.
                            t_idx = idx // (grid_spatial * grid_spatial)  # which temporal patch block
                            rem = idx % (grid_spatial * grid_spatial)
                            r_idx = rem // grid_spatial  # row in the patch grid
                            c_idx = rem % grid_spatial   # column in the patch grid
                            
                            # Convert grid indices to pixel indices in the full volume.
                            t_start = t_idx * tubelet_size
                            t_end = t_start + tubelet_size
                            
                            r_start = r_idx * patch_size
                            r_end = r_start + patch_size
                            
                            c_start = c_idx * patch_size
                            c_end = c_start + patch_size
                            
                            # Set the corresponding block in the volume to 1.
                            vol[t_start:t_end, r_start:r_end, c_start:c_end] = 1  # or use 255 if desired
                        
                        volumes.append(vol)
                    
                    return volumes

                # Step 1. Forward
                #DEBUG: MAY30 Patchify/Unpatchify Round Trip Check
                img = clips[0:1]  # single sample
                patches = patchify_image(img, patch_size)
                recon_img = unpatchify_image_from_full(patches, patch_size, tubelet_size, num_frames, in_chans, crop_size)
                logger.info(f"[DEBUG] patchify→unpatchify diff: {(recon_img - img).abs().mean()}")

                loss_mae, loss_spec, loss_reg = 0., 0., 0.
                imgs_full = None 
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    c_hat, z = forward_context(clips)
                    loss_mae = loss_fn(c_hat, clips)  # mae prediction loss
                    pstd_z = reg_var_fn(z)
                    loss_reg += torch.mean(F.relu(1. - pstd_z)) 
                     
                # ---- Spectral Loss calculation -periodical -full tokens ----
                spec_loss_active = spectral_coeff > 0 and epoch >= min_epoch_for_spectral and global_step % spec_loss_every_n_iter == 0
                if spec_loss_active:
                    with torch.no_grad():
                        ctxt_tokens, tgt_tokens = forward_context_full(clips, z)
                        enc_mask  = masks_enc[0].long()
                        pred_mask = masks_pred[0].long()
                        B, _, D = ctxt_tokens.shape
                        L_total = enc_mask.shape[1] + pred_mask.shape[1]

                        tokens_full = torch.zeros(B, L_total, D, device=ctxt_tokens.device, dtype=ctxt_tokens.dtype)
                        tokens_full.scatter_(1, enc_mask.unsqueeze(-1).expand_as(ctxt_tokens), ctxt_tokens)
                        tokens_full.scatter_(1, pred_mask.unsqueeze(-1).expand_as(tgt_tokens), tgt_tokens)

                        imgs_full = reconstruct_image_full([tokens_full])  # list of [B, C, T, H, W]

                    # spectral loss summed over the batch
                    for i in range(len(imgs_full)):
                        img_rec = imgs_full[i]  # [B, C, T, H, W]
                        img_gt  = clips          # [B, C, T, H, W]
                        for c in range(img_rec.shape[1]): #sum over channels separately for all the batch samples
                            loss_spec += spectral_loss_images(img_rec[:, c], img_gt[:, c], mode='complex')

                    # if loss_spec.item() > 1e3:
                    #     logger.warning(f"[SpectralLoss Warning] High loss_spec={loss_spec.item():.2e} at step {global_step}")

                    loss_spec /= len(imgs_full) * img_rec.shape[1]

                # Accumulate loss before stepping optimizer
                loss = (loss_mae + reg_coeff * loss_reg + spectral_coeff_eff * loss_spec) / accumulation_steps  # Normalize loss

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    if (itr + 1) % accumulation_steps == 0:  # Only unscale when we're going to step
                        scaler.unscale_(optimizer)
                        if (epoch >= warmup) and (clip_grad is not None):
                            _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                            _pred_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                        torch.cuda.synchronize()
                        loss = AllReduce.apply(loss)  # Average loss across GPUs  
                else:
                    loss.backward()
                    if (itr + 1) % accumulation_steps == 0:  # Only when we're going to step
                        if (epoch >= warmup) and (clip_grad is not None):
                            _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                            _pred_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
                        optimizer.step()
                        torch.cuda.synchronize()
                        loss = AllReduce.apply(loss)  # Average loss across GPUs
                
                # GU_DEBUG: May30 - 
                ema_beta = 0.95
                global_grad_enc= ema_beta * global_grad_enc + (1 - ema_beta) * _enc_norm
                logger.info(f"[DEBUG] Encoder grad norm EMA: {global_grad_enc:.4f}")
                logger.info(f"[DEBUG] MAE Loss: {loss:.4f}")

                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(decoder.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)                
                optim_stats = adamw_logger(optimizer)

                # GU_DEBUG: May30 - Condensed version
                if rank == 10:
                    verbose_grad_debug = False  # Set to True for full param scan

                    def print_grad_info(module, name, keys_to_log=None, max_blocks=[0, -1]):
                        logged = set()
                        for pname, param in module.named_parameters():
                            log_this = False
                            # Always show selected block indices or top-level names
                            if keys_to_log and any(key in pname for key in keys_to_log):
                                log_this = True
                            if any(f".blocks.{i}." in pname for i in max_blocks):
                                log_this = True
                            if "proj" in pname or "norm" in pname:
                                log_this = True
                            if verbose_grad_debug:
                                log_this = True

                            if log_this and pname not in logged:
                                logged.add(pname)
                                if param.grad is None:
                                    logger.warning(f"[DEBUG] {name} param {pname} has no grad!")
                                else:
                                    logger.info(f"[DEBUG] {name} param {pname} grad norm: {param.grad.norm().item():.4e}")

                    print_grad_info(encoder, "Encoder", keys_to_log=["patch_embed", "pos_embed"])
                    print_grad_info(decoder, "Decoder", keys_to_log=["predictor_proj", "predictor_norm", "mask_tokens"])

                
                if (itr + 1) % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # Efficient way to clear gradients
                #optimizer.zero_grad()  # Only zero gradients after step

                #gpu_memory_alloc = torch.cuda.max_memory_allocated() / 1024.0 ** 2 # **Monitor Memory & GPU Utilization**
                # logger.info(f"GPU Memory Allocated: {gpu_memory_alloc:.2f} MB") # **Monitor Memory & GPU Utilization**
    
                # VISUALIZATION: Reconstruct images & masks to visualize at every write_img_freq epochs
                if (itr == 0) and (epoch % write_img_freq == 0):
                    if imgs_full is not None:
                        imgs_vis = imgs_full[0][0]  #imgs_full: list of [B, C, T, H, W] → take first sample [C, T, H, W]
                        # Save each channel of reconstructed volume as a separate .nii.gz file
                        for c in range(imgs_vis.shape[0]):  # iterate over channels
                            recon_vol = imgs_vis[c].cpu().detach().float().numpy()  # shape: [T, H, W]
                            nib.save(nib.Nifti1Image(recon_vol, affine=np.eye(4)),
                                    f'ZReconstructed_full_volume_c{c}-epoch{epoch}.nii.gz')
                    # Reconstruct mosaic image: predicted patches + original pathces
                    imgs = reconstruct_image(c_hat, clips)
                    imgs_vis = imgs[0]  # [C, T, H, W]
                    logger.info(f"[DEBUG] recon image mean: {imgs_vis.mean().item():.4f}") #DEBUG: May30
                    for c in range(imgs_vis.shape[0]):  # iterate over channels
                            recon_vol = imgs_vis[c].cpu().detach().float().numpy()  # shape: [T, H, W]
                            nib.save(nib.Nifti1Image(recon_vol, affine=np.eye(4)),
                                    f'ZReconstructed_mosaic_volume_c{c}-epoch{epoch}.nii.gz')

                    # Save binary mask volumes
                    binary_volumes = reconstruct_mask_volume(
                        masks_pred, patch_size, tubelet_size, num_frames, crop_size
                    )
                    for i, vol in enumerate(binary_volumes):
                        vol_np = vol.cpu().detach().numpy().astype(np.uint8)
                        nib.save(nib.Nifti1Image(vol_np, affine=np.eye(4)),
                                f'ZMask_volume_{i}-epoch{epoch}.nii.gz')

                return (
                    float(loss) * accumulation_steps,  # Restore original loss scale
                    float(loss_mae),
                    float(loss_spec),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                    spectral_coeff_eff,
                )
            
            (loss, loss_mae, loss_spec, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,spectral_coeff_eff,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            mae_loss_meter.update(loss_mae)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)
            
            global global_step
            spec_loss_active = spectral_coeff > 0 and epoch >= min_epoch_for_spectral and global_step % spec_loss_every_n_iter == 0
            global_step += 1 #for warmup of spectral loss coef

            # Release memory
            del clips
            torch.cuda.empty_cache()

            # -- Logging
            def log_stats():
                # Handle inactive spectral loss logging
                loss_spec_val = loss_spec if spec_loss_active else float('-1.0')

                # CSV log (writes everything — still useful to see 'nan' in skipped steps)
                csv_logger.log(
                    epoch,
                    itr,
                    loss,
                    loss_mae,
                    loss_spec_val,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms
                )

                
                # Tensorboard logging
                # log_writer.add_scalar('train/loss', loss, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/loss_mae', loss_mae, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/loss_reg', loss_reg, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/global_norm', grad_stats.global_norm, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/pred_global_norm', grad_stats_pred.global_norm, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/gpu_etime_ms', gpu_etime_ms, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/iter_elapsed_time_ms', iter_elapsed_time_ms, (epoch * ipe) + itr)
                # log_writer.add_scalar('train/memory', torch.cuda.max_memory_allocated() / 1024.0**2, (epoch * ipe) + itr)
                # log_writer.flush()
                
                
                # Wandb logging
                if run != None and rank == 0:
                    log_dict = {
                        'train/loss': loss,
                        'train/loss_mae': loss_mae,
                        'train/loss_reg': loss_reg,
                        'train/global_norm': grad_stats.global_norm,
                        'train/pred_global_norm': grad_stats_pred.global_norm,
                        'train/gpu_etime_ms': gpu_etime_ms,
                        'train/iter_elapsed_time_ms': iter_elapsed_time_ms,
                        'train/memory': torch.cuda.max_memory_allocated() / 1024.0**2,
                        'train/lr': _new_lr,
                        'train/wd': _new_wd,
                    }

                    if spec_loss_active:
                        log_dict['train/loss_spec'] = loss_spec
                        log_dict['debug/spectral_coeff_eff'] = spectral_coeff_eff
                        log_dict['debug/spec_active_step'] = global_step


                    run.log(log_dict, step=global_step)
                    # run.log({
                    #         'train/loss': loss,
                    #         'train/loss_mae': loss_mae,
                    #         'train/loss_reg': loss_reg,
                    #         'train/global_norm': grad_stats.global_norm,
                    #         'train/pred_global_norm': grad_stats_pred.global_norm,
                    #         'train/gpu_etime_ms': gpu_etime_ms,
                    #         'train/iter_elapsed_time_ms': iter_elapsed_time_ms,
                    #         'train/memory': torch.cuda.max_memory_allocated() / 1024.0**2,
                    #         'train/lr': _new_lr,
                    #         'train/wd': _new_wd
                    #     })
                
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
                           mae_loss_meter.avg,
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
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))    
            assert not np.isnan(loss), 'loss is nan'

            torch.cuda.empty_cache()

        # -- Save Checkpoint
        logger.info('--- Epoch avg. loss %.3f ---' % loss_meter.avg)
        
        # -- Save checkpoint or last epoch
       # if ((itr == 0) and (epoch % checkpoint_freq == 0) or (epoch == (num_epochs - 1))) and log_dir != None:
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

        torch.cuda.empty_cache()

    if run != None:
        run.finish()
