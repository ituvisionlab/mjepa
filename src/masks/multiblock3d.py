# mjepa: A 3D MRI self-supervised learning framework based on a modified V-JEPA
# Copyright (c) 2024–2025 [Gozde Unal, NYU]
#
# This file is based on an earlier version of code from:
# V-JEPA (https://github.com/facebookresearch/v-jepa)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This codebase has been significantly modified for use in medical imaging and 3D MRI.
# All modifications are licensed under the original MIT license (or the applicable license).

import math
import time
from multiprocessing import Value
from logging import getLogger
import torch
from torch.utils.data import get_worker_info

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    #GU_
    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):

        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        #mask_start_time = time.time()  # **Measure Mask Generation**

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        # mask_end_time = time.time()  # **Measure Mask Generation**
        # mask_gen_time = mask_end_time - mask_start_time # **Measure Mask Generation**
        # logger.info(f"MaskCollator Time: {mask_gen_time:.4f} sec") # **Measure Mask Generation**

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=0.5, #1 GU_Debug, changed this to a ratio of masked patches
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
      
        #seed = self.step() #GU_
        # seed = torch.initial_seed() # Obtain the initial seed for the current worker
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)
        # Create a shared counter BEFORE seeding.
        # In distributed settings each DataLoader worker has its own info.
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            self.base_seed = (torch.initial_seed() + worker_id) % (2**32)
            self.local_g = torch.Generator()
            self.local_g.manual_seed(self.base_seed)

        # A simple Python counter for per-call variation.
        #self.call_count = 0


        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = int(npred * self.height * self.width * self.duration) #GU_DEBUG, use npred as a ratio of masking
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
    

    # def step(self):
    #     i = self._itr_counter
    #     with i.get_lock():
    #         i.value += 1
    #         v = i.value
    #     return v
    # def step(self):
    #     # Increment the counter so that each call produces a different offset.
    #     with self._itr_counter.get_lock():
    #         self._itr_counter.value += 1
    #         # Return the new seed as base_seed plus the counter offset.
    #         return self.base_seed + self._itr_counter.value
    # def _create_local_generator(self):
    #     """Create a new generator for each call based on the worker-specific base seed and a per-call counter."""
    #     self.call_count += 1
    #     new_seed = (self.base_seed + self.call_count) % (2**32)
    #     local_g = torch.Generator()
    #     local_g.manual_seed(new_seed)
    #     return local_g
    
    # def _create_local_generator(self):
    #     """Create a persistent generator per worker, updated every batch."""
    #     if not hasattr(self, "local_g"):
    #         self.local_g = torch.Generator()
    #         self.local_g.manual_seed(self.base_seed)
    #     self.call_count += 1
    #     self.local_g.manual_seed((self.base_seed + self.call_count) % (2**32))
    #     return self.local_g


    # def _sample_block_size(
    #     self,
    #     generator,
    #     temporal_scale,
    #     spatial_scale,
    #     aspect_ratio_scale
    # ):
        # -- Sample temporal block mask scale
        # _rand = torch.rand(1, generator=generator).item()
        # min_t, max_t = temporal_scale
        # temporal_mask_scale = min_t + _rand * (max_t - min_t)
        # t = max(1, int(self.duration * temporal_mask_scale))

        # # -- Sample spatial block mask scale
        # _rand = torch.rand(1, generator=generator).item()
        # min_s, max_s = spatial_scale
        # spatial_mask_scale = min_s + _rand * (max_s - min_s)
        # spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # # -- Sample block aspect-ratio
        # _rand = torch.rand(1, generator=generator).item()
        # min_ar, max_ar = aspect_ratio_scale
        # aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # # -- Compute block height and width (given scale and aspect-ratio)
        # h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        # w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        # h = min(h, self.height)
        # w = min(w, self.width)

        # #GU_Debug: Override the above
        # h = w = spatial_scale[0]
        # t = temporal_scale[0]
        # return (t, h, w)

    def _sample_block_size(self):
        """Block sampling based on fixed token sizes."""
        h = w = int(self.spatial_pred_mask_scale[0])  # Ensure integer values
        t = int(self.temporal_pred_mask_scale[0])  # Ensure integer values
        return (t, h, w)
    
     #GU_
    # def _sample_block_mask(self, b_size, generator):
    #     t, h, w = b_size
    #     top = torch.randint(0, self.height - h + 1, (1,), generator=generator).item()
    #     left = torch.randint(0, self.width - w + 1, (1,), generator=generator).item()
    #     start = torch.randint(0, self.duration - t + 1, (1,), generator=generator).item()

    #     mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
    #     mask[start:start+t, top:top+h, left:left+w] = 0

    #     if self.max_context_duration < self.duration:
    #         mask[self.max_context_duration:, :, :] = 0

    #     return mask

    def _sample_block_mask(self, b_size):
        """Generate mask blocks efficiently without looping over npred."""
        t, h, w = map(int, b_size)  # Convert all values to int
        num_masks = self.npred

        # Sample top-left corner locations
        top = torch.randint(0, self.height - h + 1, (num_masks,))
        left = torch.randint(0, self.width - w + 1, (num_masks,))
        start = torch.randint(0, self.duration - t + 1, (num_masks,))
        # top = torch.randint(0, self.height - h + 1, (num_masks,), generator=self.local_g)
        # left = torch.randint(0, self.width - w + 1, (num_masks,), generator=self.local_g)
        # start = torch.randint(0, self.duration - t + 1, (num_masks,), generator=self.local_g)

        # Initialize full mask (1s everywhere)
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)

        # Use advanced indexing to set mask blocks to 0
        for i in range(num_masks):
            mask[start[i]:start[i]+t, top[i]:top[i]+h, left[i]:left[i]+w] = 0

        # Context mask will only span first X frames
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask
    # def _sample_block_mask(self, b_size):
    #     t, h, w = b_size
    #     top = torch.randint(0, self.height - h + 1, (1,))
    #     left = torch.randint(0, self.width - w + 1, (1,))
    #     start = torch.randint(0, self.duration - t + 1, (1,))

    #     mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
    #     mask[start:start+t, top:top+h, left:left+w] = 0

    #     # Context mask will only span the first X frames
    #     # (X=self.max_context_frames)
    #     if self.max_context_duration < self.duration:
    #         mask[self.max_context_duration:, :, :] = 0

    #     # --
    #     return mask

    def _sample_block_mask1(self, b_size):
        """
        Efficiently generate mask blocks using full vectorization.
        """
        t, h, w = map(int, b_size)  # Ensure integer values

        num_masks = self.npred  # Number of masked blocks

        # Sample top-left corner locations **all at once**
        start = torch.randint(0, max(1, self.duration - t + 1), (num_masks,))
        top = torch.randint(0, max(1, self.height - h + 1), (num_masks,))
        left = torch.randint(0, max(1, self.width - w + 1), (num_masks,))

        # Create an empty mask of ones
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int8, device='cpu')

        # Generate full indices for masked regions (vectorized)
        t_indices = (start[:, None] + torch.arange(t)).flatten()
        h_indices = (top[:, None] + torch.arange(h)).flatten()
        w_indices = (left[:, None] + torch.arange(w)).flatten()

        # Ensure indices remain within valid bounds
        t_indices = t_indices.clamp(0, self.duration - 1)
        h_indices = h_indices.clamp(0, self.height - 1)
        w_indices = w_indices.clamp(0, self.width - 1)

        # Use tensor indexing to set mask blocks to **zero** (masked)
        mask[t_indices, h_indices, w_indices] = 0

        # Restrict context mask span
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        # Update generator seed based on the worker's initial seed plus a per-call offset.
        #new_seed = self.step()
        #self.g.manual_seed(new_seed)

        # Create a new generator for this call.
        #local_g = self._create_local_generator()

        b_size = self._sample_block_size()
        # p_size = self._sample_block_size(
        #     generator=local_g,
        #     temporal_scale=self.temporal_pred_mask_scale,
        #     spatial_scale=self.spatial_pred_mask_scale,
        #     aspect_ratio_scale=self.aspect_ratio,
        # )

        #mask_start_time = time.time()  # **Start timing mask generation

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        # for _ in range(batch_size): #GU_

        empty_context = True
        while empty_context:
            mask_e = self._sample_block_mask(b_size)  # Vectorized sampling
            mask_e = mask_e.flatten()

            # To-DO: try fully vectorized sampling
            # mask_e = self._sample_block_mask1(b_size).flatten()

            #mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
            #for _ in range(self.npred):
            #    mask_e *= self._sample_block_mask(p_size, generator=local_g) #GU_ Too much slowed down! All random functions will use the same generator g, ensuring consistent and random mask generation within each worker.
            # mask_e = mask_e.flatten()

            mask_p = torch.argwhere(mask_e == 0).squeeze()
            mask_e = torch.nonzero(mask_e).squeeze()

            empty_context = len(mask_e) == 0
            if not empty_context:
                min_keep_pred = min(min_keep_pred, len(mask_p))
                min_keep_enc = min(min_keep_enc, len(mask_e))
                collated_masks_pred.append(mask_p)
                collated_masks_enc.append(mask_e)
        
        # mask_end_time = time.time()  # **End timing mask generation
        # mask_gen_time = mask_end_time - mask_start_time # **End timing mask generation
        # print(f"MaskGenerator Time: {mask_gen_time:.4f} sec") # **End timing mask generation

        #GU_ replicate the masks along the batchsize
        collated_masks_enc *= batch_size
        collated_masks_pred *= batch_size

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        # # GU_Debug Print mask shapes
        # print(f"Mask Encoder Shape: {collated_masks_enc[0].shape}")
        # print(f"Mask Predictor Shape: {collated_masks_pred[0].shape}")

        return collated_masks_enc, collated_masks_pred

