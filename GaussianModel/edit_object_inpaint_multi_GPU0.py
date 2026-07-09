# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import torchvision
from PIL import Image
import cv2
import numpy as np
import lpips

from utils.loss_utils import masked_l1_loss
from random import randint

from render import feature_to_rgb, visualize_obj
from edit_object_removal import points_inside_convex_hull
from os import makedirs

# -------------------------
# Utility functions
# -------------------------
def mask_to_bbox(mask):
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

# -------------------------
# Finetune (per-process)
# -------------------------
def finetune_inpaint_worker(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg):
    """
    rank: process rank (0..world_size-1)
    world_size: total processes / GPUs
    args: parsed CLI args (namespace)
    model_cfg / opt_cfg / pipeline_cfg are extracted objects created in main and passed here (lightweight)
    """

    # -------------------------
    # init distributed process group
    # -------------------------
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Re-seed to get different randomness per rank (but reproducible)
    seed = args.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np_random = __import__('numpy').random
    np_random.seed(seed)

    # -------------------------
    # load models & data (each process instantiates its own copy)
    # -------------------------
    gaussians = GaussianModel(model_cfg.sh_degree)
    scene = Scene(model_cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    num_classes = model_cfg.num_classes

    # build classifier (used for mask computation); move to device
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).to(device)
    # load checkpoint to CPU then map to device
    ckpt_path = os.path.join(model_cfg.model_path, "point_cloud", "iteration_"+str(scene.loaded_iter), "classifier.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        # strip module. if necessary
        try:
            classifier.load_state_dict(state)
        except RuntimeError:
            new_state = {}
            for k,v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            classifier.load_state_dict(new_state)
    classifier = classifier.to(device)
    # classifier is only used in inference to get masks, so no DDP needed for it specifically.
    # If you want classifier gradients synchronized, wrap it into DDP as well.

    # Background tensor
    bg_color = [1,1,1] if model_cfg.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Prepare LPIPS (VGG) and wrap with DDP to ensure it's on the device and (optionally) synchronized
    LPIPS = lpips.LPIPS(net='vgg').to(device)
    for p in LPIPS.parameters():
        p.requires_grad = False
    # LPIPS does not have trainable params typically; DDP is harmless but not necessary.
    # If you prefer wrap:
    # LPIPS = DDP(LPIPS, device_ids=[rank])

    # compute per-process number of iterations
    total_iters = args.finetune_iteration
    iters_per_proc = total_iters // world_size
    # If total_iters not divisible, make last rank take the remainder
    if rank == world_size - 1:
        iters_per_proc += total_iters % world_size

    # -------------------------
    # compute mask3d once (same across ranks) using classifier
    # -------------------------
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1).to(device))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[args.select_obj_id, :, :] > args.removal_thresh
        mask3d = mask.any(dim=0).squeeze()
        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach().to(device), mask3d, outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d.to(device), mask3d_convex)
        mask3d = mask3d.float()[:, None, None]

    # set up gaussians for inpainting (note: this mutates local copy)
    gaussians.inpaint_setup(opt_cfg, mask3d)

    # optimizer is inside gaussians (as in your original code)
    LPIPS_loss_fn = LPIPS

    # Use progress bar only on rank 0
    progress = None
    if rank == 0:
        progress = tqdm(total=iters_per_proc, desc=f"Rank {rank} Finetuning")

    # main local finetune loop
    for local_iter in range(iters_per_proc):
        # choose viewpoint randomly (you used randint earlier)
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipeline_cfg, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg.get("viewspace_points", None)
        visibility_filter = render_pkg.get("visibility_filter", None)
        radii = render_pkg.get("radii", None)
        objects = render_pkg.get("render_object", None)

        mask2d = viewpoint_cam.objects > 128
        gt_image = viewpoint_cam.original_image.to(device)
        Ll1 = masked_l1_loss(image.to(device), gt_image, ~mask2d)

        bbox = mask_to_bbox(mask2d)
        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)
        K = 2
        rendering_patches = divide_into_patches(cropped_image[None, ...], K)
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
        # LPIPS expects inputs in [-1,1]
        lpips_loss = LPIPS_loss_fn(rendering_patches.squeeze().to(device)*2-1, gt_patches.squeeze().to(device)*2-1).mean()

        loss = (1.0 - opt_cfg.lambda_dssim) * Ll1 + opt_cfg.lambda_dssim * lpips_loss
        loss.backward()

        with torch.no_grad():
            # local update logic (same as original)
            if local_iter < 5000 and visibility_filter is not None and radii is not None:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if local_iter % 300 == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(opt_cfg.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if rank == 0 and (local_iter % max(1, iters_per_proc//10) == 0):
            progress.set_postfix({"Loss": f"{loss:.7f}"})
            progress.update(max(1, iters_per_proc//10))

    if progress is not None:
        progress.close()

    # --- Optional: synchronize gaussians parameters across ranks via averaging ---
    # If you want final gaussians parameters to be averaged across ranks, you can implement an all-reduce
    # For example (pseudocode):
    # for name, tensor in gaussians.named_parameters():  # if GaussianModel exposes parameters iterable
    #     dist.all_reduce(tensor.data, op=dist.ReduceOp.SUM)
    #     tensor.data /= world_size
    #
    # But this requires GaussianModel to expose its parameters as tensors. If it does not, you'll need to
    # implement a manual serialization / averaging (e.g., save checkpoints per rank and average them on CPU).
    #
    # Here we do not implement automatic parameter sync per step. We simply let each rank update its local copy,
    # and we choose to save the result from rank 0 (you may prefer to save all ranks or average them).

    if rank == 0:
        # save gaussians from rank 0 to the expected place
        point_cloud_path = os.path.join(model_cfg.model_path, "point_cloud_object_inpaint/iteration_{}".format(args.finetune_iteration-1))
        makedirs(point_cloud_path, exist_ok=True)
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_rank0.ply"))
        # You may also save gaussians internal checkpoint if supported:
        try:
            gaussians.save(os.path.join(point_cloud_path, "gaussians_rank0.pth"))
        except Exception:
            pass

    # cleanup
    dist.barrier()
    dist.destroy_process_group()


# -------------------------
# Render set (single-process / can be called from rank 0)
# -------------------------
def render_set_single(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits, dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    size = (gt.shape[-1]*5, gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt, rgb, gt_obj, pred_obj, render_obj]).astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()

# -------------------------
# Entry point: spawn processes
# -------------------------
def main_worker_spawn(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg):
    # wrapper for mp.spawn
    finetune_inpaint_worker(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg)

def main():
    parser = ArgumentParser(description="Multi-GPU inpaint finetune")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json")
    parser.add_argument("--finetune_iteration", type=int, default=10000)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=12355)
    parser.add_argument("--seed", type=int, default=42)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read config file
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {args.config_file} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", args.finetune_iteration)

    # initialize RNGs
    safe_state(args.quiet)

    # world size
    world_size = args.world_size
    if world_size <= 1:
        print("Single GPU or CPU mode. Running single-process.")
        # run single-process original inpaint path for backward compatibility
        # We call the worker with rank 0 and world_size 1
        finetune_inpaint_worker(0, 1, args, model.extract(args), opt.extract(args), pipeline.extract(args))


    else:
        # spawn one process per GPU (assumes CUDA_VISIBLE_DEVICES set)
        print(f"Spawning {world_size} processes for multi-GPU finetune.")
        mp.spawn(main_worker_spawn, args=(world_size, args, model.extract(args), opt.extract(args), pipeline.extract(args)), nprocs=world_size, join=True)

    # After finishing training, rank 0 can render results (optional)
    # If you want to render using the final gaussians from rank 0, you can load them and call render_set_single.
    print("Multi-GPU finetune finished.")

if __name__ == "__main__":
    main()
    if rank == 0 and local_iter == 0:
        print("render_pkg keys:", render_pkg.keys())
