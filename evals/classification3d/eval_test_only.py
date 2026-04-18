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
    #world_size, rank = init_distributed()
    world_size, rank = 1, 0
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        local_rank = -1
    else:
        # works on slurm and also torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

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
        checkpoint_key="encoder"
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
    for p in encoder.parameters(): p.requires_grad = False
    encoder.eval()
    test_loader, test_sampler = make_dataloader(
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


    # NOTE: Esra: I'm commenting out the following to make the code compatible with UHEM.
    """
    encoder = DistributedDataParallel(encoder, static_graph=True, gradient_as_bucket_view=True) #GU_Debug
    """
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
    """
    test_acc, test_f1, test_auc, test_precision, test_recall = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_clips,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            scaler=None,
            optimizer=None,
            scheduler=None,
            wd_scheduler=None,
            data_loader=test_loader,
            data_sampler=test_sampler,
            use_bfloat16=use_bfloat16,
            log_writer=log_writer,
            epoch=None,
            eval_freq=val_eval_freq,
            rank=rank,
            run=run,
            num_classes=num_classes,
            warmup=warmup,
            clip_grad_encoder=clip_grad_encoder,
            clip_grad_classifier=clip_grad_classifier)

    if rank == 0:
        logger.info(f'FINAL: test acc: {test_acc:.3f}% recall: {test_recall/100:.3f} '
                    f'precision: {test_precision/100:.3f} f1: {test_f1/100:.3f} AUC: {test_auc/100:.3f}')
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

    encoder.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    ipe = len(data_loader)
    if eval_freq is None or eval_freq > ipe:
        eval_freq = 1
 
    loader = iter(data_loader)
    data_sampler.set_epoch(0)
    
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
            #top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)
            
            # Collect results for AUC during validation
            if not training:
                all_outputs.append(F.softmax(outputs, dim=1).detach().cpu())

                all_labels.append(labels.detach().cpu())

        
        torch.cuda.empty_cache()

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
                
    return val_acc, val_f1, val_auc, val_precision, val_recall

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
        pretrained_dict = checkpoint['encoder']
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
    pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
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
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1.0,1.0), #(0.75, 4/3),
    random_resize_scale=(0.9, 1.0),
    rot_degree=0,
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
    checkpoint_key="encoder"
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
