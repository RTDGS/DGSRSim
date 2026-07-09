# edit_object_inpaint_multi_GPU_fixed.py
# Comprehensive multi-GPU inpaint (fixed memory leaks, DDP spawn)
# Keep Scene, GaussianModel, render as blackbox but ensure safe device handling.

import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import torchvision
from PIL import Image
import cv2
import lpips

from utils.loss_utils import masked_l1_loss
from random import randint
from os import makedirs

# SciPy Delaunay for convex hull check (CPU)
from scipy.spatial import Delaunay

# -------------
# Helper: robust points_inside_convex_hull that works on CPU and returns mask on device
# -------------
def points_inside_convex_hull_cpu(point_cloud: torch.Tensor, mask: torch.Tensor, remove_outliers=True, outlier_factor=1.0, device=None):
    """
    point_cloud: (N,3) tensor (can be on CUDA) -> will be moved to CPU for scipy
    mask: (N,) boolean mask (can be on CUDA)
    returns: boolean tensor (N,) on `device` (or same device as point_cloud if device None)
    """
    if device is None:
        device = point_cloud.device

    # move to cpu numpy
    pc_cpu = point_cloud.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy().astype(bool)

    if mask_cpu.sum() == 0:
        return torch.zeros(point_cloud.shape[0], dtype=torch.bool, device=device)

    masked_points = pc_cpu[mask_cpu]

    if remove_outliers and masked_points.shape[0] >= 4:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # need at least 4 non-coplanar points for 3D Delaunay; fallback to 2D if needed:
    try:
        if filtered_masked_points.shape[0] < 4:
            # fallback: return convex hull of masked xy-projection if enough points
            # We'll treat insufficient points as "no convex hull" -> return all False
            return torch.zeros(point_cloud.shape[0], dtype=torch.bool, device=device)
        delaunay = Delaunay(filtered_masked_points)
        inside = delaunay.find_simplex(pc_cpu) >= 0
    except Exception:
        # if Delaunay fails, return all False rather than crash
        inside = np.zeros(pc_cpu.shape[0], dtype=bool)

    return torch.tensor(inside, dtype=torch.bool, device=device)

# -------------
# Training worker (one process per GPU)
# -------------
def finetune_inpaint_worker(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg):
    """
    rank: 0..world_size-1
    Each process binds to GPU `rank` (assumes CUDA_VISIBLE_DEVICES has been set accordingly).
    """
    # ------------------------
    # init distributed group
    # ------------------------
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # seeds per rank
    seed = args.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------
    # instantiate scene & gaussians (per-process copy)
    # ------------------------
    gaussians = GaussianModel(model_cfg.sh_degree)
    scene = Scene(model_cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    num_classes = model_cfg.num_classes

    # classifier: used only for mask computation (inference). Put on device and set eval/no_grad.
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).to(device)
    ckpt_path = os.path.join(model_cfg.model_path, "point_cloud", "iteration_"+str(scene.loaded_iter), "classifier.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        try:
            classifier.load_state_dict(state)
        except Exception:
            # try stripping "module." prefix
            new_state = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state.items()}
            classifier.load_state_dict(new_state)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # background tensor
    bg_color = [1,1,1] if model_cfg.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # LPIPS: freeze parameters but compute gradients w.r.t. inputs (renderings).
    LPIPS = lpips.LPIPS(net='vgg').to(device)
    LPIPS.eval()
    for p in LPIPS.parameters():
        p.requires_grad = False

    # compute mask3d once (same across ranks) using classifier under no_grad to avoid grad mem
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1).to(device))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        # args.select_obj_id might be a list or single int
        if isinstance(args.select_obj_id, (list, tuple)):
            sel_idx = args.select_obj_id[0]
        else:
            sel_idx = args.select_obj_id
        mask = prob_obj3d[sel_idx, :, :] > args.removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        # get convex hull mask via CPU helper; return to `device`
        mask3d_convex = points_inside_convex_hull_cpu(gaussians._xyz.detach(), mask3d.detach(), outlier_factor=1.0, device=device)
        mask3d = torch.logical_or(mask3d.to(device), mask3d_convex).float()[:, None, None].to(device)

    # setup inpaint (mutates local gaussians)
    gaussians.inpaint_setup(opt_cfg, mask3d)

    total_iters = args.finetune_iteration
    # split iterations across ranks (last rank takes remainder)
    base_iters = total_iters // world_size
    iters_for_rank = base_iters + (total_iters % world_size if rank == world_size - 1 else 0)

    # progress (only rank 0 shows progress)
    progress = None
    if rank == 0:
        progress = tqdm(total=iters_for_rank, desc=f"Rank {rank} finetune")

    # Tuneables
    EMPTY_CACHE_EVERY = 50  # call torch.cuda.empty_cache every N iterations (safe)
    LOG_EVERY = max(1, iters_for_rank // 50)

    # Main loop
    for local_iter in range(iters_for_rank):
        # choose viewpoint
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render: IMPORTANT - render should produce tensors on the correct device already.
        # We will keep the rendering as part of the graph (to allow gradients to flow to gaussians),
        # but will avoid keeping references after optimizer step.
        render_pkg = render(viewpoint_cam, gaussians, pipeline_cfg, background)

        # Unpack safely
        image = render_pkg["render"]  # expected to be tensor, requires_grad True for backprop
        viewspace_point_tensor = render_pkg.get("viewspace_points", None)
        visibility_filter = render_pkg.get("visibility_filter", None)
        radii = render_pkg.get("radii", None)
        rendering_obj = render_pkg.get("render_object", None)

        # Move GT to device
        gt_image = viewpoint_cam.original_image.to(device)
        mask2d = (viewpoint_cam.objects > 128)  # keep as CPU tensor for bounding box; not part of graph

        # Compute L1 masked loss (ensure tensors on same device)
        # masked_l1_loss expects image and gt on same device
        Ll1 = masked_l1_loss(image.to(device), gt_image, ~mask2d)  # masked_l1_loss should handle devices

        # Crop to bbox
        # convert mask2d to tensor on device for bbox ops
        mask2d_tensor = mask2d.to(device)
        # mask_to_bbox
        rows = torch.any(mask2d_tensor, dim=1)
        cols = torch.any(mask2d_tensor, dim=0)
        if rows.any() and cols.any():
            ymin, ymax = torch.where(rows)[0][[0, -1]]
            xmin, xmax = torch.where(cols)[0][[0, -1]]
            bbox = (xmin.item(), ymin.item(), xmax.item(), ymax.item())
            # cropping - image and gt_image shapes: (C,H,W)
            cropped_image = image[:, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            cropped_gt_image = gt_image[:, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        else:
            # fallback to full image if mask empty
            cropped_image = image
            cropped_gt_image = gt_image

        # Divide patches (ensure batch dim)
        def divide_into_patches_local(img, K):
            # img: (C,H,W) -> return (num_patches, C, ph, pw)
            C, H, W = img.shape
            ph, pw = H // K, W // K
            if ph == 0 or pw == 0:
                # fallback
                return img.unsqueeze(0)
            img_b = img.unsqueeze(0)  # 1, C, H, W
            patches = torch.nn.functional.unfold(img_b, (ph, pw), stride=(ph, pw))  # 1, C*ph*pw, L
            patches = patches.view(1, C, ph, pw, -1).permute(0, 4, 1, 2, 3)  # 1,L,C,ph,pw -> drop first dim
            patches = patches.squeeze(0)  # L, C, ph, pw
            return patches

        K = 2
        rendering_patches = divide_into_patches_local(cropped_image, K).to(device)
        gt_patches = divide_into_patches_local(cropped_gt_image, K).to(device)

        # LPIPS expects inputs in [-1,1], batch shape (N,C,H,W)
        # LPIPS model parameters are frozen, but gradients should flow to inputs (rendering_patches)
        if rendering_patches.ndim == 3:
            rendering_patches = rendering_patches.unsqueeze(0)
            gt_patches = gt_patches.unsqueeze(0)
        # clamp / ensure float
        rendering_patches = rendering_patches.float()
        gt_patches = gt_patches.float()

        # compute LPIPS (no grad for LPIPS params but keep graph so grads flow to inputs)
        lpips_loss = LPIPS(rendering_patches * 2.0 - 1.0, gt_patches * 2.0 - 1.0).mean()

        # Total loss (lpips contributes gradients to rendering)
        loss = (1.0 - opt_cfg.lambda_dssim) * Ll1 + opt_cfg.lambda_dssim * lpips_loss

        # Backward and optimizer step
        gaussians.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # optional: clip grads if you have a stable way to access tensors
        try:
            torch.nn.utils.clip_grad_norm_(gaussians.optimizer.param_groups[0]['params'], 1.0)
        except Exception:
            # if gaussians.optimizer.param_groups structure is unknown, skip
            pass
        gaussians.optimizer.step()

        # After step: update densify/prune stats exactly as before but ensure we detach tensors used for stats
        with torch.no_grad():
            if local_iter < 5000 and visibility_filter is not None and radii is not None:
                # Ensure assignments use detached tensors
                vf = visibility_filter
                rr = radii
                gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rr[vf])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if local_iter % 300 == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(opt_cfg.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

        # Logging & cleanup
        if rank == 0 and progress is not None and (local_iter % LOG_EVERY == 0):
            progress.set_postfix({"loss": float(loss.detach().cpu().item())})
            progress.update(min(LOG_EVERY, iters_for_rank - local_iter))

        # free large tensors and detach references to allow GC
        del render_pkg
        # detach and delete intermediate tensors
        try:
            del image, rendering_obj, viewspace_point_tensor, visibility_filter, radii
            del cropped_image, cropped_gt_image, rendering_patches, gt_patches, lpips_loss
            del Ll1, loss
        except Exception:
            pass

        # periodic explicit cache clear
        if (local_iter + 1) % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

    if progress is not None:
        progress.close()

    # Optional: rank 0 saves result
    if rank == 0:
        point_cloud_path = os.path.join(model_cfg.model_path, "point_cloud_object_inpaint/iteration_{}".format(args.finetune_iteration-1))
        makedirs(point_cloud_path, exist_ok=True)
        try:
            gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_rank0.ply"))
            gaussians.save(os.path.join(point_cloud_path, "gaussians_rank0.pth"))
        except Exception:
            # if save methods differ, fallback to save_ply only
            try:
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_rank0.ply"))
            except Exception:
                print("Warning: saving gaussians failed on rank 0.")

    # Barrier & cleanup
    dist.barrier()
    dist.destroy_process_group()

# -------------
# Optional rendering helper for rank0 after training (single-process rendering)
# -------------
def render_set_single(model_path, name, iteration, views, gaussians, pipeline, background, classifier, device):
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

    out_path = os.path.join(render_path[:-8], 'concat')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    size = (gt.shape[-1] * 5, gt.shape[-2])
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

# -------------
# Entrypoint: spawn processes
# -------------
def main_worker_spawn(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg):
    finetune_inpaint_worker(rank, world_size, args, model_cfg, opt_cfg, pipeline_cfg)

def main():
    parser = ArgumentParser(description="Multi-GPU inpaint finetune - fixed")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json")
    parser.add_argument("--finetune_iteration", type=int, default=10000)
    parser.add_argument("--world_size", type=int, default=(torch.cuda.device_count() if torch.cuda.is_available() else 1))
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
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        return

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", args.finetune_iteration)

    # Initialize RNG
    safe_state(args.quiet)

    world_size = int(args.world_size)
    if world_size <= 1:
        print("Single GPU / CPU mode: running single-process worker.")
        finetune_inpaint_worker(0, 1, args, model.extract(args), opt.extract(args), pipeline.extract(args))
    else:
        print(f"Spawning {world_size} processes for multi-GPU finetune.")
        mp.spawn(main_worker_spawn, args=(world_size, args, model.extract(args), opt.extract(args), pipeline.extract(args)), nprocs=world_size, join=True)

    print("Multi-GPU finetune finished.")

if __name__ == "__main__":
    main()