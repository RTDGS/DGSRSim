# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import json

import cv2
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull, Delaunay
from render import feature_to_rgb, visualize_obj


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Robust convex-hull inclusion test.

    Accepts:
      point_cloud: torch.Tensor (N,3) or (3,N) (or numpy equivalents)
      mask:        torch.Tensor/Numpy (N,), (N,1), (N,1,1) or scalar (degenerate)

    Returns:
      torch.BoolTensor (N,) on the same device as point_cloud.
    """
    device = point_cloud.device if torch.is_tensor(point_cloud) else torch.device("cpu")

    # ---- normalize points to numpy (N,3) ----
    if torch.is_tensor(point_cloud):
        pts = point_cloud.detach().float().cpu().numpy()
    else:
        pts = np.asarray(point_cloud, dtype=np.float32)

    if pts.ndim != 2:
        raise ValueError(f"point_cloud must be 2D, got shape {pts.shape}")

    if pts.shape[1] == 3:
        pass
    elif pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    else:
        raise ValueError(f"point_cloud must be (N,3) or (3,N), got shape {pts.shape}")

    N = pts.shape[0]

    # ---- normalize mask to numpy ----
    if torch.is_tensor(mask):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)

    m = np.squeeze(m)

    # If mask becomes scalar, treat as degenerate -> return all False (no convex extension)
    if m.ndim == 0:
        inside = np.zeros((N,), dtype=bool)
        return torch.from_numpy(inside).to(device=device)

    if m.ndim != 1:
        # be permissive: try flatten
        m = m.reshape(-1)

    if m.shape[0] != N:
        # cannot match points -> return all False (avoid crash)
        inside = np.zeros((N,), dtype=bool)
        return torch.from_numpy(inside).to(device=device)

    m = (m > 0.5)

    masked_points = pts[m]
    if masked_points.shape[0] < 4:
        inside = np.zeros((N,), dtype=bool)
        return torch.from_numpy(inside).to(device=device)

    filtered_masked_points = masked_points
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - outlier_factor * IQR
        upper = Q3 + outlier_factor * IQR
        outlier_mask = (masked_points < lower) | (masked_points > upper)
        keep = ~np.any(outlier_mask, axis=1)
        filtered_masked_points = masked_points[keep]

    if filtered_masked_points.shape[0] < 4:
        inside = np.zeros((N,), dtype=bool)
        return torch.from_numpy(inside).to(device=device)

    try:
        hull = ConvexHull(filtered_masked_points)
        delaunay = Delaunay(filtered_masked_points[hull.vertices])
        inside = delaunay.find_simplex(pts) >= 0
    except Exception:
        inside = np.zeros((N,), dtype=bool)

    return torch.from_numpy(inside).to(device=device)


def removal_setup(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        # IMPORTANT: mask3d is (N,) bool, gaussians._xyz can be (N,3) or (3,N); function is robust now.
        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), mask3d, outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d, mask3d_convex)

        mask3d = mask3d.float()[:, None, None]

    # fix some gaussians
    gaussians.removal_setup(opt, mask3d)

    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    os.makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
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
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    size = (gt.shape[-1] * 5, gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt, rgb, gt_obj, pred_obj, render_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()


def removal(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, opt: OptimizationParams, select_obj_id: int, removal_thresh: float):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ", num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "classifier.pth")))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. remove selected object
    gaussians = removal_setup(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh)

    # 3. render new result
    scene = Scene(dataset, gaussians, load_iteration='_object_removal/iteration_' + str(scene.loaded_iter), shuffle=False)
    with torch.no_grad():
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true", default="")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    removal(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh)
