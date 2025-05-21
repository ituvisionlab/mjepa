# mjepa/dino: A 3D MRI self-supervised learning framework based on a modified V-JEPA
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

from src.datasets.data_manager import init_data
#from src.masks.random_tube import MaskCollator as TubeMaskCollator
#from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
#from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.dino.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.dino.transforms import make_dino_transforms # New augmentation function
from app.dino.utils import DinoCenterManager
from app.dino.utils import cosine_similarity, dino_debug_dashboard

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
periodic_ckpt_save_freq = 25 
# --

global_step = 0 # global counter for debug

_GLOBAL_SEED = 0
#np.random.seed(_GLOBAL_SEED)
#torch.manual_seed(_GLOBAL_SEED)
#torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

def main(args, resume_preempt=False, log_dir="./logs/evals", run=None):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    global global_step
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

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    embed_dim = cfgs_model.get('embed_dim',768) #default value for vit_base
    uniform_power = cfgs_model.get('uniform_power', True)

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
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
    random_blur = cfgs_data_aug.get('random_blur', [0.5, 1.5])
    intensity_gamma = cfgs_data_aug.get('intensity_gamma', 0.2)
    local_crop_ratio = cfgs_data_aug.get('local_crop_ratio', 0.85) 
    max_offset_fraction = cfgs_data_aug.get('max_offset_fraction', 0.1) 
    use_aa = cfgs_data_aug.get('auto_augment', False)
    num_global_views = cfgs_data_aug.get('num_global_views', 2)
    num_local_views = cfgs_data_aug.get('num_local_views', 6)

    # -- LOSS
    cfgs_loss = args.get('loss')
    reg_coeff = cfgs_loss.get('reg_coeff', 0.0)
    temperature_student = cfgs_loss.get('temperature_student', 0.1)
    temperature_teacher = cfgs_loss.get('temperature_teacher', 0.03)
    momentum = cfgs_loss.get('momentum', 0.9)
    alpha_vcr = cfgs_loss.get('alpha_vcr',1.0)
    beta_vcr = cfgs_loss.get('beta_vcr',0.1) #0.4

    #loss_exp = cfgs_loss.get('loss_exp')

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
    dino_ckpt_folder = cfgs_meta.get("ckpt_folder", "src/models/pretrained_weights")
    
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
        load_path = os.path.join(dino_ckpt_folder, r_file) if r_file is not None else None
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
            ('%.5f', 'loss-dino'),
            #('%.5f', 'loss-jepa'),
            ('%.5f', 'reg-loss'),
            ('%.5f', 'enc-grad-norm'),
            ('%d', 'gpu-time(ms)'),
            ('%d', 'wall-time(ms)'),
        )
    else:
        tb_rank_folder = None
        log_writer = None
        
        log_file = None
        csv_logger = None
    
    
    # -- init model: DINO: encoder=STUDENT, target_encoder: TEACHER
    #encoder, predictor = init_video_model(
    encoder  = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        in_chans=in_chans,
        use_sdpa=use_sdpa,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    # if mask_type == 'multiblock3d':
    #     logger.info('Initializing basic multi-block mask')
    #     mask_collator = MB3DMaskCollator(
    #         crop_size=crop_size,
    #         num_frames=num_frames,
    #         patch_size=patch_size,
    #         tubelet_size=tubelet_size,
    #         cfgs_mask=cfgs_mask)
    # else:
    #     logger.info('Initializing random tube mask')
    #     mask_collator = TubeMaskCollator(
    #         crop_size=crop_size,
    #         num_frames=num_frames,
    #         patch_size=patch_size,
    #         tubelet_size=tubelet_size,
    #         cfgs_mask=cfgs_mask)
        
    transform = make_dino_transforms(
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        rot_degree = rot_degree,
        auto_augment=use_aa,
        crop_size=crop_size,
        local_crop_ratio=local_crop_ratio,
        max_offset_fraction = max_offset_fraction,
        intensity_gamma=intensity_gamma,
        random_bias=random_bias,
        random_noise=random_noise,
        random_blur=random_blur)

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
         # collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=csv_folder if log_resource_util_data else None,
         vol_type="dino")
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
        # predictor=predictor,
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
    # predictor = DistributedDataParallel(predictor, static_graph=True, gradient_as_bucket_view=True)
    target_encoder = DistributedDataParallel(target_encoder, gradient_as_bucket_view=True)
    for p in target_encoder.parameters():
        p.requires_grad = False

    #student_encoder = DistributedDataParallel(student_encoder, device_ids=[device], find_unused_parameters=True)
    #teacher_encoder = DistributedDataParallel(teacher_encoder, device_ids=[device], find_unused_parameters=True)


    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch=0
    # -- load training checkpoint
    # if load_model or os.path.exists(load_path):
    if load_model and os.path.exists(load_path):
        (
            encoder,
            # predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            # predictor=predictor,
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
            # 'predictor': predictor.state_dict(),
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
    # Subtract a moving average of teacher outputs (i.e. the "center") from the teacher logits before computing the softmax.
    center_manager = DinoCenterManager(embed_dim, device, momentum)
    global last_center
    last_center = center_manager.get().detach().clone()

    # --- Teacher temperature warm-up schedule
    warmup_epochs = 30
    delta_temp=0.04
    total_epochs = num_epochs
    teacher_temp_schedule = np.concatenate([
        np.linspace(temperature_teacher, temperature_teacher+delta_temp, warmup_epochs),
        np.ones(total_epochs - warmup_epochs) * temperature_teacher+delta_temp
    ])

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
        dino_loss_meter = AverageMeter()  # Track DINO loss
        # jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            iter_start_time = time.time()
           
            try:
                udata  = next(loader) 
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                torch.cuda.empty_cache()
                
                loader = iter(unsupervised_loader) #resets the loader iterator again
                udata = next(loader)             

            clips = udata[0] #dino preprocessing gives multiview of data, a list of tensors, one per augmented view

            def train_step():
                global global_step
                global last_center 

                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --
                
                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each view.
                    Only global views are passed to the teacher.
                    """
                    with torch.no_grad():
                        h = []
                        for view_batch in c:
                            view_batch = view_batch.to(device)  # [B, C, D, H, W]
                            h.append(target_encoder(view_batch))  # pass as tensor
                        return h


                def forward_context(c):
                    """
                    Returns two lists of tensors of shape [B, N, D] from student encoder:
                    one list for global views, one for local views.
                    """
                    noise_std=0.01
                    global_views = []
                    for i, view_batch in enumerate(c[0:num_global_views]):
                        view_batch = view_batch.to(device)
                        # Inject noise into only one global view for student to introduce asymmetry
                        if i == 0:
                            view_batch = view_batch + torch.randn_like(view_batch) * noise_std
                        global_views.append(encoder(view_batch))

                    local_views = []
                    for view_batch in c[num_global_views:]:
                        view_batch = view_batch.to(device)
                        out = encoder(view_batch)
                        out = F.dropout(out, p=0.4, training=True)  #Add noise to student's views!
                        local_views.append(out)
                        #local_views.append(encoder(view_batch))

                    return global_views, local_views

                def dino_loss_fn(student_output, teacher_output, center_manager):
                    """
                    Computes DINO loss with centering.
                    Inputs:
                        student_output: [B, N, D]
                        teacher_output: [B, N, D]
                        center: [1, 1, D] — running average buffer
                    Returns:
                        loss (scalar), updated center (tensor)
                    """
                    center = center_manager.get()
                    temp_teacher = teacher_temp_schedule[epoch]

                    # Centering the teacher output [B, N, D]
                    teacher_logits = (teacher_output - center) / temp_teacher
                    student_logits = student_output / temperature_student

                    # Softmax along feature dim (D)
                    teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
                    student_log_probs = F.log_softmax(student_logits, dim=-1)
                    
                    # KL divergence per token: [B, N]
                    loss_per_token = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-6) - student_log_probs), dim=-1)
                    loss = loss_per_token.mean() # Average over all tokens and batch
 
                    center_manager.update(teacher_output)  

                    return loss

                
                # def reg_fn(z):
                #     return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # def reg_fn(z_list): #just computes the variance loss term
                #     """
                #     Encourages std deviation across batch tokens per feature dimension to be ≥ 1.
                #     Returns average std over dimensions.
                #     """
                #     reg = 0.
                #     for z in z_list:
                #         # z: [B, N, D]
                #         z = z.view(-1, z.shape[-1])  # flatten: [B*N, D]
                #         std = torch.sqrt(z.var(dim=0) + 1e-4)  # [D]
                #         reg += std.mean()  # average over D
                #     return reg / len(z_list)

                def reg_fn(z_list):
                    """
                    Computes variance and covariance regularization:
                    - Penalizes std dev < 1 (ReLU(1 - std))
                    - Penalizes off-diagonal covariance
                    Args:
                        z_list: list of [B, N, D] tensors
                    Returns:
                        scalar regularization loss
                    """
                    var_loss = 0.0
                    cov_loss = 0.0

                    for z in z_list:
                        B, N, D = z.shape
                        # -- Variance across tokens per feature dim
                        std = torch.sqrt(z.var(dim=1) + 1e-4)  # [B, D]
                        std_mean = std.mean(dim=0)            # [D]
                        var_loss += F.relu(1. - std_mean).mean()

                        # -- Covariance across batch-token features
                        z_flat = z.view(-1, D)  # [B*N, D]
                        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
                        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
                        off_diag = cov - torch.diag(torch.diag(cov))
                        cov_loss += (off_diag ** 2).sum() / D

                    var_loss /= len(z_list)
                    cov_loss /= len(z_list)

                    return alpha_vcr * var_loss + beta_vcr * cov_loss

                # Step 1. Forward
                loss_dino, loss_reg = 0., 0.
                # loss_jepa = 0.

                #forward_start_time = time.time() # **Measure Forward Pass Time**
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # Compute teacher and student features
                    h_teacher = forward_target(clips[0:num_global_views])  # Only global views are processed into teacher
                    z_g_student, z_l_student = forward_context(clips)  # All views go into student 

                    # Cosine similarity
                    cosine_sim = cosine_similarity(z_g_student[0], h_teacher[0])
                                                   
                    # Compute DINO loss
                    #num_loss_terms = 0
                    for student_out in z_g_student + z_l_student:      # 6 student views
                        for teacher_out in h_teacher:        # 2 teacher views
                            loss_view = dino_loss_fn(student_out, teacher_out, center_manager)
                            loss_dino += loss_view
                            #num_loss_terms += 1
                    loss_dino /= (len(z_g_student) + len(z_l_student)) * len(h_teacher)
                    #loss_dino /= num_loss_terms
                    loss_reg = reg_fn(z_g_student + z_l_student)

                # Accumulate loss before stepping optimizer
                loss = (loss_dino + reg_coeff * loss_reg) / accumulation_steps
                # loss = (loss_dino +loss_jepa + reg_coeff * loss_reg) / accumulation_steps
  
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
                            #_pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                        #torch.cuda.synchronize()
                        #loss = AllReduce.apply(loss)  # Average loss across GPUs  
                else:
                    loss.backward()
                    if (itr + 1) % accumulation_steps == 0:  # Only when we're going to step
                        if (epoch > warmup) and (clip_grad is not None):
                            _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                            #_pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                        optimizer.step()
                        #torch.cuda.synchronize()
                        #loss = AllReduce.apply(loss)  # Average loss across GPUs
                
                # backward_end_time = time.time() # **Measure Backward Pass + Optimizer Step**
                # backward_time = backward_end_time - backward_start_time # **Measure Backward Pass + Optimizer Step**
                # logger.info(f"Backward Pass Time: {backward_time:.4f} sec") # **Measure Backward Pass + Optimizer Step**

                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                optim_stats = adamw_logger(optimizer)

                if (itr + 1) % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # Efficient way to clear gradients
                #optimizer.zero_grad()  # Only zero gradients after step

                # Step 3. momentum update of teacher (target encoder)
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                debug_metrics = {}  # default value
                if global_step % 50 == 0 and rank == 0:
                    debug_metrics = dino_debug_dashboard(
                        global_step=global_step,
                        logger=logger,
                        z_g_student=z_g_student,
                        h_teacher=h_teacher,
                        center=center_manager.get(),
                        prev_center=last_center.clone(),
                        student_input_0=z_g_student[0],
                        student_input_1=z_g_student[1],
                        temperature_teacher=temperature_teacher,
                        abort_on_collapse=False,
                    )

                last_center = center_manager.get().detach().clone()  # store current for next delta calc

                return (
                    float(loss) * accumulation_steps,  # Restore original loss scale
                    float(loss_dino), 
                    #float(loss_jepa),
                    float(loss_reg),
                    float(cosine_sim),
                    debug_metrics,
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    optim_stats,
                )
            (loss, loss_dino, loss_reg, sim_score, debug_metrics, _new_lr, _new_wd, grad_stats, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - iter_start_time) * 1000.
            
            if run is not None and global_step % 50 == 0 and rank == 0:
                run.log(debug_metrics, step=global_step) # wandb logging
            
            # Update loss meters
            loss_meter.update(loss)
            dino_loss_meter.update(loss_dino)
            # jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)
 
            input_var_total = 0.0
            input_var_min_total = float('inf')
            for clip in clips:
                reshaped = clip.view(clip.shape[0], -1)         # [B, C*D*H*W]
                var_per_sample = reshaped.var(dim=1)            # [B]
                input_var_total += var_per_sample.mean()
                input_var_min_total = min(input_var_min_total, var_per_sample.min())
            # # 
            # input_var = float(AllReduce.apply(input_var_total / len(clips)))
            # input_var_min = float(AllReduce.apply(input_var_min_total))
            # Convert scalars to GPU tensors before AllReduce
            input_var_tensor = torch.tensor(input_var_total / len(clips), device=device)
            input_var = float(AllReduce.apply(input_var_tensor))

            input_var_min_tensor = torch.tensor(input_var_min_total, device=device)
            input_var_min = float(AllReduce.apply(input_var_min_tensor))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
         
           
            gpu_memory_alloc = torch.cuda.max_memory_allocated() / 1024.0 ** 2 # **Monitor Memory & GPU Utilization**
            # print(gpu_memory_alloc)
            # logger.info(f"GPU Memory Allocated: {gpu_memory_alloc:.2f} MB") # **Monitor Memory & GPU Utilization**

            #Increment global_step
            global_step += 1 

            # Release memory
            del clips
            torch.cuda.empty_cache()

            # -- Logging
            def log_stats():
                if rank == 0:
                    center_values = center_manager.get().detach().cpu()  # shape [1, 1, D]
                    center_mean = center_values.mean().item()
                    center_std = center_values.std().item()
                    center_min = center_values.min().item()
                    center_max = center_values.max().item()
                else:
                    center_mean = center_std = center_min = center_max = 0.0  # default for non-zero ranks

                csv_logger.log(
                    epoch,
                    itr,
                    loss,
                    loss_dino,
                    loss_reg,
                    center_mean,
                    grad_stats.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)                
                
                # Wandb logging
                if run != None and rank == 0:
                    run.log({
                            'train/loss': loss,
                            'train/loss_dino': loss_dino,
                            #'train/loss_jepa': loss_jepa,
                            'train/loss_reg': loss_reg,
                            'train/sim_score': sim_score,
                            'train/global_norm': grad_stats.global_norm,
                            'train/gpu_etime_ms': gpu_etime_ms,
                            'train/iter_elapsed_time_ms': iter_elapsed_time_ms,
                            'train/memory': gpu_memory_alloc,
                            'train/lr': _new_lr,
                            'train/wd': _new_wd,
                            'center/mean': center_mean, # Center tracking
                            'center/std': center_std,
                            'center/min': center_min,
                            'center/max': center_max,
                        })
                
            def info_stats():
                
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | d%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        #'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch, itr,
                           loss_meter.avg,
                           dino_loss_meter.avg,
                           # jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           #'[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           gpu_memory_alloc,
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
            
            if log_dir != None:
                log_stats()
                
            info_stats()
                
            assert not np.isnan(loss), 'loss is nan'

            torch.cuda.empty_cache()

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

        torch.cuda.empty_cache()

        # SUBMIT A Classifier Evaluation Periodically
        #if epoch % 150 == 0:
        #    subprocess.call(['sbatch', './test.sh']) 

    if run != None:
        run.finish()
