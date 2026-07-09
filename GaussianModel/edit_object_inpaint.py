# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import json
import os
from argparse import ArgumentParser
from os import makedirs
from random import randint

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from edit_object_removal import points_inside_convex_hull
from gaussian_renderer import GaussianModel, render
from render import feature_to_rgb, visualize_obj
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import masked_l1_loss, knn_laplacian_loss


def finetune_inpaint(
    opt,
    model_path,
    loaded_iter,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    selected_obj_ids,
    cameras_extent,
    removal_thresh,
    finetune_iteration,
    lambda_geo=0.05,
    k_lap=8,
    use_densify=False,
):
    # 1) find 3D gaussians corresponding to the selected object id
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        obj_mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(
            gaussians._xyz.detach(), obj_mask3d, outlier_factor=1.0
        )
        obj_mask3d = torch.logical_or(obj_mask3d, mask3d_convex)
        obj_mask3d_setup = obj_mask3d.float()[:, None, None]

    # 2) only finetune the selected region according to GaussianModel.inpaint_setup
    gaussians.inpaint_setup(opt, obj_mask3d_setup)

    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")

    for it in range(iterations):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # current config uses images_inpaint_unseen as dataset.images,
        # so viewpoint_cam.original_image is already the no-object background supervision image.
        bg_gt_image = viewpoint_cam.original_image.cuda()

        # foreground mask from object annotations: True means object region.
        fg_mask = viewpoint_cam.objects > 128
        bg_mask = ~fg_mask

        # Stable background reconstruction loss
        l_bg = masked_l1_loss(image, bg_gt_image, bg_mask)

        # Local geometric smoothing over the editable region
        l_lap = knn_laplacian_loss(gaussians._xyz, obj_mask3d, k=k_lap)

        loss = l_bg + lambda_geo * l_lap
        loss.backward()


        with torch.no_grad():
            if use_densify and it < min(iterations, 2000):
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if it % 300 == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold
                    )

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if it % 10 == 0:
            progress_bar.set_postfix(
                {
                    "L_bg": f"{l_bg.item():.7f}",
                    "L_lap": f"{l_lap.item():.7f}",
                    "Loss": f"{loss.item():.7f}",
                }
            )
            progress_bar.update(10)
    progress_bar.close()

    point_cloud_path = os.path.join(
        model_path, f"point_cloud_object_inpaint/iteration_{finetune_iteration - 1}"
    )
    makedirs(point_cloud_path, exist_ok=True)
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
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, "{0:05d}".format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, "{0:05d}".format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, "{0:05d}".format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))

    out_path = os.path.join(render_path[:-8], 'concat')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX')
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



def inpaint(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    opt: OptimizationParams,
    select_obj_id: int,
    removal_thresh: float,
    finetune_iteration: int,
    lambda_geo: float,
    k_lap: int,
    use_densify: bool,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ", num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(
        torch.load(
            os.path.join(
                dataset.model_path,
                "point_cloud",
                "iteration_" + str(scene.loaded_iter),
                "classifier.pth",
            )
        )
    )
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = finetune_inpaint(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        classifier,
        select_obj_id,
        scene.cameras_extent,
        removal_thresh,
        finetune_iteration,
        lambda_geo=lambda_geo,
        k_lap=k_lap,
        use_densify=use_densify,
    )

    # reset for result rendering: use original images / masks for visualization
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/iteration_' + str(finetune_iteration - 1), shuffle=False)
    with torch.no_grad():
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/object_removal/bear.json",
        help="Path to the configuration file",
    )

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

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
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.finetune_iteration = config.get("finetune_iteration", 10000)
    args.lambda_geo = config.get("lambda_geo", 0.05)
    args.k_lap = config.get("k_lap", 8)
    args.use_densify = config.get("use_densify", False)

    safe_state(args.quiet)

    inpaint(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,
        args.removal_thresh,
        args.finetune_iteration,
        args.lambda_geo,
        args.k_lap,
        args.use_densify,
    )
