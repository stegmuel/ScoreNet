from source.datasets.bracs_base_dataset import BracsBaseDataset
from source.datasets.bach_base_dataset import BachBaseDataset
from torch.utils.data import DataLoader
from source.utils.constants import *
from einops import rearrange, repeat
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil, sqrt
from PIL import Image
import seaborn as sns
from glob import glob
import numpy as np
import imagesize
import logging
import random
import torch
import ast
import cv2


def get_logger(logfile):
    """
    Create and return a logger to store ans display the training statuses.
    :param logfile: location where to write the log outputs.
    :return: new logger.
    """
    # Instantiate a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def correct_predictions(batch_outputs, batch_labels):
    """
    Computes the number of correct predictions in a given batch.
    :param batch_outputs: batch of outputs predicted by the model.
    :param batch_labels: ground truth values for the batch.
    :return: number of correct predictions.
    """
    batch_predictions = torch.argmax(batch_outputs, dim=1)
    return batch_predictions.eq(batch_labels.squeeze()).sum().float().item()


def get_data_loaders(args):
    if args.dataset == 'bach':
        # Instantiate the datasets
        train_dataset = BachBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BACH,
            queries=args.train_queries if isinstance(args.train_queries, list) else ast.literal_eval(args.train_queries),
            phase='train',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            class_dict=CLASS_DICT_BACH,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
        )
        valid_dataset = BachBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BACH,
            queries=args.valid_queries if isinstance(args.valid_queries, list) else ast.literal_eval(args.valid_queries),
            phase='valid',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            class_dict=CLASS_DICT_BACH,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
        )
        test_dataset = BachBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BACH,
            queries=args.test_queries if isinstance(args.test_queries, list) else ast.literal_eval(args.test_queries),
            phase='test',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            class_dict=CLASS_DICT_BACH,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
        )
    else:
        # Instantiate the datasets
        train_dataset = BracsBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BRACS_TROI,
            queries=args.train_queries if isinstance(args.train_queries, list) else ast.literal_eval(args.train_queries),
            phase='train',
            class_dict=CLASS_DICT_BRACS_TROI_PREVIOUS,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        valid_dataset = BracsBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BRACS_TROI,
            queries=args.valid_queries if isinstance(args.valid_queries, list) else ast.literal_eval(args.valid_queries),
            phase='valid',
            class_dict=CLASS_DICT_BRACS_TROI_PREVIOUS,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        test_dataset = BracsBaseDataset(
            seed=args.seed,
            phase_dict=PHASE_DICT_BRACS_TROI,
            queries=args.test_queries if isinstance(args.test_queries, list) else ast.literal_eval(args.test_queries),
            phase='test',
            class_dict=CLASS_DICT_BRACS_TROI_PREVIOUS,
            patch_dim=args.large_patch_size,
            thumbnail_resolution=args.small_patch_size,
            k=args.n_patches,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader, test_loader


def saliency_bbox(img, lam, size):
    h, w = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, w)
    bby1 = np.clip(y - cut_h // 2, 0, h)
    bbx2 = np.clip(x + cut_w // 2, 0, w)
    bby2 = np.clip(y + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def scoremix_bbox_loop(lam, saliency_maps):
    # Map lambda to an integer number of patches
    n_h = min([saliency_map.shape[-2] for saliency_map in saliency_maps])
    n_w = min([saliency_map.shape[-1] for saliency_map in saliency_maps])
    n_tot = n_h * n_w
    n_tot_lam = ceil(lam * n_tot)

    # Sample the number of patches in the height of the bbox
    if n_tot_lam == 1:
        n_h_lam = n_w_lam = 1
    else:
        n_h_lam = min(np.random.randint(low=1, high=ceil(sqrt(n_tot_lam)), size=1)[0], n_h)
        n_w_lam = min(ceil(n_tot_lam / n_h_lam), n_w)

    # Compute the value of lambda for each image
    n_tot_lam = n_w_lam * n_h_lam
    lam = [1. - n_tot_lam / saliency_map.shape[-2] / saliency_map.shape[-1] for saliency_map in saliency_maps]

    # Find the most and less salient regions of each image
    b_c_h_w_indices_max, b_c_h_w_indices_min = [], []
    for saliency_map in saliency_maps:
        kernel = torch.ones([1, 1, n_h_lam, n_w_lam]).to(saliency_map.device)
        saliency_map = rearrange(saliency_map, 'h w -> 1 1 h w')
        salient_region = F.conv2d(saliency_map, kernel)
        salient_region = rearrange(salient_region, '1 1 h w -> h w')
        s_h, s_w = salient_region.shape
        salient_region = rearrange(salient_region, 'h w -> (h w)')

        # Get the positive masks
        discriminative_dist = salient_region - salient_region.min()
        try:
            dist = torch.distributions.Categorical(discriminative_dist)
        except ValueError:
            discriminative_dist = torch.ones_like(discriminative_dist)
            dist = torch.distributions.Categorical(discriminative_dist)
        max_indices = dist.sample()
        # max_indices = torch.argmax(salient_region, dim=-1)
        h_starts = max_indices // s_w
        h_range = torch.arange(0, n_h_lam).to(saliency_map.device)
        h_indices = h_starts + h_range
        w_starts = max_indices % s_w
        w_range = torch.arange(0, n_w_lam).to(saliency_map.device)
        w_indices = w_starts + w_range
        c_indices = torch.arange(0, 3).to(saliency_map.device)

        h_indices = repeat(h_indices, 'h -> c h w', w=n_w_lam, c=3)
        w_indices = repeat(w_indices, 'w -> c h w', h=n_h_lam, c=3)
        c_indices = repeat(c_indices, 'c -> c h w', h=n_h_lam, w=n_w_lam)
        c_h_w_indices = torch.stack([c_indices, h_indices, w_indices], dim=0)
        c_h_w_indices = rearrange(c_h_w_indices, 'd c h w -> (c h w) d')
        c_h_w_indices_max = list(map(lambda x: tuple(x), c_h_w_indices.T.tolist()))
        b_c_h_w_indices_max.append(c_h_w_indices_max)

        # Get the positive masks
        non_discriminative_dist = 1 - discriminative_dist
        non_discriminative_dist = non_discriminative_dist - non_discriminative_dist.min()
        try:
            dist = torch.distributions.Categorical(non_discriminative_dist)
        except ValueError:
            non_discriminative_dist = torch.ones_like(non_discriminative_dist)
            dist = torch.distributions.Categorical(non_discriminative_dist)
        min_indices = dist.sample()
        # min_indices = torch.argmin(salient_region, dim=-1)
        h_starts = min_indices // s_w
        h_range = torch.arange(0, n_h_lam).to(saliency_map.device)
        h_indices = h_starts + h_range
        w_starts = min_indices % s_w
        w_range = torch.arange(0, n_w_lam).to(saliency_map.device)
        w_indices = w_starts + w_range

        h_indices = repeat(h_indices, 'h -> c h w', w=n_w_lam, c=3)
        w_indices = repeat(w_indices, 'w -> c h w', h=n_h_lam, c=3)
        c_h_w_indices = torch.stack([c_indices, h_indices, w_indices], dim=0)
        c_h_w_indices = rearrange(c_h_w_indices, 'd c h w -> (c h w) d')
        c_h_w_indices_min = list(map(lambda x: tuple(x), c_h_w_indices.T.tolist()))
        b_c_h_w_indices_min.append(c_h_w_indices_min)

    # Return the masks and the corrected lambda value
    return b_c_h_w_indices_max, b_c_h_w_indices_min, lam


def scoremix_bbox(lam, saliency_maps, index):
    # Map lambda to an integer number of patches
    n_h, n_w = saliency_maps.shape[-2], saliency_maps.shape[-1]
    n_tot = n_h * n_w
    n_tot_lam = ceil(lam * n_tot)

    # Sample the number of patches in the height of the bbox
    if n_tot_lam == 1:
        n_h_lam = n_w_lam = 1
    else:
        n_h_lam = min(np.random.randint(low=1, high=ceil(sqrt(n_tot_lam)), size=1)[0], n_h)
        n_w_lam = min(ceil(n_tot_lam / n_h_lam), n_w)

    # Re-compute the value of lambda
    n_tot_lam = n_h_lam * n_w_lam
    lam = 1. - n_tot_lam / n_tot

    # Find the most and less salient regions of each image
    kernel = torch.ones([1, 1, n_h_lam, n_w_lam]).to(saliency_maps.device)
    salient_regions = F.conv2d(saliency_maps.unsqueeze(1), kernel)
    b, _, s_h, s_w = salient_regions.shape
    salient_regions = rearrange(salient_regions, 'b 1 h w -> b (h w)')

    # Get the positive masks
    # max_indices = torch.argmax(salient_regions, dim=-1)
    discriminative_dist = salient_regions - salient_regions.min(dim=-1, keepdim=True).values
    try:
        dist = torch.distributions.Categorical(discriminative_dist)
    except ValueError:
        discriminative_dist = torch.ones_like(discriminative_dist)
        dist = torch.distributions.Categorical(discriminative_dist)
    max_indices = dist.sample()

    h_starts = max_indices // s_w
    h_range = torch.arange(0, n_h_lam).to(saliency_maps.device)
    h_indices = h_starts[:, None] + h_range[None, :]
    w_starts = max_indices % s_w
    w_range = torch.arange(0, n_w_lam).to(saliency_maps.device)
    w_indices = w_starts[:, None] + w_range[None, :]
    c_indices = torch.arange(0, 3).to(saliency_maps.device)
    # b_indices = torch.arange(0, b).to(saliency_maps.device)
    b_indices = index.to(saliency_maps.device)

    h_indices = repeat(h_indices, 'b h -> b c h w', w=n_w_lam, c=3)
    w_indices = repeat(w_indices, 'b w -> b c h w', h=n_h_lam, c=3)
    c_indices = repeat(c_indices, 'c -> b c h w', b=b, h=n_h_lam, w=n_w_lam)
    b_indices = repeat(b_indices, 'b -> b c h w', h=n_h_lam, w=n_w_lam, c=3)
    b_h_w_indices = torch.stack([b_indices, c_indices, h_indices, w_indices], dim=0)
    b_h_w_indices = rearrange(b_h_w_indices, 'd b c h w -> (b c h w) d')
    b_h_w_indices_max = list(map(lambda x: tuple(x), b_h_w_indices.T.tolist()))

    # Get the positive masks
    # min_indices = torch.argmin(salient_regions, dim=-1)
    non_discriminative_dist = 1 - discriminative_dist
    non_discriminative_dist = non_discriminative_dist - non_discriminative_dist.min(dim=-1, keepdim=True).values
    # non_discriminative_dist = non_discriminative_dist / non_discriminative_dist.sum()
    try:
        dist = torch.distributions.Categorical(non_discriminative_dist)
    except ValueError:
        non_discriminative_dist = torch.ones_like(non_discriminative_dist)
        dist = torch.distributions.Categorical(non_discriminative_dist)
    min_indices = dist.sample()
    h_starts = min_indices // s_w
    h_range = torch.arange(0, n_h_lam).to(saliency_maps.device)
    h_indices = h_starts[:, None] + h_range[None, :]
    w_starts = min_indices % s_w
    w_range = torch.arange(0, n_w_lam).to(saliency_maps.device)
    w_indices = w_starts[:, None] + w_range[None, :]
    b_indices = torch.arange(0, b).to(saliency_maps.device)

    h_indices = repeat(h_indices, 'b h -> b c h w', w=n_w_lam, c=3)
    w_indices = repeat(w_indices, 'b w -> b c h w', h=n_h_lam, c=3)
    b_indices = repeat(b_indices, 'b -> b c h w', h=n_h_lam, w=n_w_lam, c=3)
    b_h_w_indices = torch.stack([b_indices, c_indices, h_indices, w_indices], dim=0)
    b_h_w_indices = rearrange(b_h_w_indices, 'd b c h w -> (b c h w) d')
    b_h_w_indices_min = list(map(lambda x: tuple(x), b_h_w_indices.T.tolist()))

    # Return the masks and the corrected lambda value
    return b_h_w_indices_max, b_h_w_indices_min, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    import os
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
