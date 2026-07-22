raise SystemExit("Legacy training snapshot disabled. Use GaussianModel/train.py.")

# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import sys
import json
import uuid
import torch

from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import (
    DEFAULT_SEMANTIC_CLASSES,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)

import wandb


def masked_l1_rgb_loss(pred, target, mask, eps=1e-6):
    """Mean RGB L1 over a binary pixel mask."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    pixel_l1 = (pred - target).abs().mean(dim=0, keepdim=True)
    return (pixel_l1 * mask).sum() / mask.sum().clamp_min(eps)


def masked_mean_abs(value, mask, eps=1e-6):
    """Mean absolute response over a binary pixel mask."""
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value.abs() * mask).sum() / mask.sum().clamp_min(eps)


def get_object_alpha(render_pkg):
    """Return object alpha if the renderer exposes it; otherwise return None."""
    for key in ("render_object_alpha", "object_alpha", "alpha_object", "render_alpha_object"):
        if key in render_pkg:
            return render_pkg[key]
    return None


def local_geometry_consistency_loss(xyz, weights=None, k=8, max_points=2048, eps=1e-6):
    """
    DGSRSim background local geometry regularizer.

    The shared Gaussian field is sampled with background-class probabilities as
    soft pair weights. The detached coordinates determine only the kNN graph;
    gradients still propagate through the selected Gaussian centers.
    """
    if xyz.shape[0] <= k:
        return xyz.new_zeros(())

    if weights is not None:
        weights = weights.detach().flatten()
        candidate_idx = torch.nonzero(weights > 0.5, as_tuple=False).flatten()
        if candidate_idx.numel() <= k:
            candidate_idx = torch.arange(xyz.shape[0], device=xyz.device)
    else:
        candidate_idx = torch.arange(xyz.shape[0], device=xyz.device)

    if candidate_idx.numel() > max_points:
        perm = torch.randperm(candidate_idx.numel(), device=xyz.device)[:max_points]
        candidate_idx = candidate_idx[perm]

    xyz_sample = xyz[candidate_idx]
    if xyz_sample.shape[0] <= k:
        return xyz.new_zeros(())

    with torch.no_grad():
        dist = torch.cdist(xyz_sample.detach(), xyz_sample.detach())
        nn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]

    neighbor_xyz = xyz_sample[nn_idx]
    pair_dist = torch.linalg.norm(xyz_sample[:, None, :] - neighbor_xyz, dim=-1)

    if weights is None:
        return pair_dist.mean()

    sampled_weights = weights[candidate_idx].to(dtype=xyz.dtype)
    pair_weights = sampled_weights[:, None] * sampled_weights[nn_idx]
    return (pair_dist * pair_weights).sum() / pair_weights.sum().clamp_min(eps)


def classifier_3d_probabilities(classifier, gaussians):
    logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
    prob_obj3d = torch.softmax(logits3d, dim=0)

    if prob_obj3d.dim() == 3 and prob_obj3d.shape[-1] == 1:
        return prob_obj3d.squeeze(-1).permute(1, 0)  # [N, C]
    return prob_obj3d.reshape(prob_obj3d.shape[0], -1).permute(1, 0)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    use_wandb,
):
    first_iter = 0
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    num_classes = dataset.num_classes
    print("Num classes: ", num_classes)

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier = classifier.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # ---------------------------------------------------------
    # DGSRSim shared-field objective:
    # L_object = L_obj-fg + lambda_alpha * L_obj-alpha
    # L_bg     = L_bg-rgb + lambda_bg_reg * L_bg-reg
    # L_total  = L_object + L_bg + lambda_sem * L_sem
    # ---------------------------------------------------------
    lambda_alpha = float(getattr(opt, "dgsrsim_lambda_alpha", 0.10))
    lambda_bg_reg = float(getattr(opt, "dgsrsim_lambda_bg_reg", 0.50))
    lambda_sem = float(getattr(opt, "dgsrsim_lambda_sem", 1.00))
    dgsrsim_background_label = int(getattr(opt, "dgsrsim_background_label", 0))
    dgsrsim_knn_k = int(getattr(opt, "dgsrsim_knn_k", 8))
    dgsrsim_reg_max_points = int(getattr(opt, "dgsrsim_reg_max_points", 2048))

    viewpoint_stack = None
    validated_semantic_views = set()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()

                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )

                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        objects = render_pkg["render_object"]  # expected [D, H, W]

        # Normalized semantic cross-entropy for shared-field ownership.
        gt_obj = viewpoint_cam.objects.cuda().long()
        view_key = id(viewpoint_cam)
        if view_key not in validated_semantic_views:
            label_min = int(gt_obj.min().item())
            label_max = int(gt_obj.max().item())
            if label_min < 0 or label_max >= num_classes:
                raise ValueError(
                    "Semantic labels must be in [0, num_classes - 1]; "
                    f"observed [{label_min}, {label_max}] with num_classes={num_classes}"
                )
            validated_semantic_views.add(view_key)
        logits = classifier(objects)
        loss_sem = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_sem = loss_sem / torch.log(
            torch.tensor(num_classes, device=loss_sem.device, dtype=loss_sem.dtype)
        )

        # ---------------------------------------------------------
        # DGSRSim object foreground supervision:
        # L_obj-fg = sum M_obj * ||I - I_obj||_1
        # ---------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        valid_mask = gt_obj >= 0
        obj_mask = valid_mask & (gt_obj != dgsrsim_background_label)
        bg_mask = valid_mask & (gt_obj == dgsrsim_background_label)

        Ll1 = l1_loss(image, gt_image)
        loss_obj_fg = masked_l1_rgb_loss(image, gt_image, obj_mask)

        # ---------------------------------------------------------
        # DGSRSim mask-outside response suppression:
        # L_obj-alpha = sum (1 - M_obj) * |alpha_obj|
        #
        # If the renderer does not expose an object-alpha map, use the summed
        # non-background class probability as a response proxy.
        # ---------------------------------------------------------
        object_alpha = get_object_alpha(render_pkg)
        if object_alpha is not None:
            loss_obj_alpha = masked_mean_abs(object_alpha, bg_mask)
        else:
            prob_2d = torch.softmax(logits, dim=0)
            if prob_2d.shape[0] > dgsrsim_background_label:
                object_response = prob_2d.sum(dim=0) - prob_2d[dgsrsim_background_label]
            else:
                object_response = prob_2d.squeeze(0)
            loss_obj_alpha = masked_mean_abs(object_response, bg_mask)

        loss_object = loss_obj_fg + lambda_alpha * loss_obj_alpha

        # ---------------------------------------------------------
        # DGSRSim background RGB supervision and local geometry regularization.
        # L_bg-rgb = sum M_bg * ||I - I_bg||_1
        # L_bg-reg = sum_i sum_j_in_N(i) ||mu_i - mu_j||_2
        # ---------------------------------------------------------
        loss_bg_rgb = masked_l1_rgb_loss(image, gt_image, bg_mask)

        if iteration % opt.reg3d_interval == 0:
            prob_obj3d = classifier_3d_probabilities(classifier, gaussians)
            if prob_obj3d.shape[1] > dgsrsim_background_label:
                bg_weights = prob_obj3d[:, dgsrsim_background_label]
            else:
                bg_weights = None
            loss_bg_reg = local_geometry_consistency_loss(
                gaussians.get_xyz,
                weights=bg_weights,
                k=dgsrsim_knn_k,
                max_points=dgsrsim_reg_max_points,
            )
        else:
            loss_bg_reg = image.new_zeros(())

        loss_bg = loss_bg_rgb + lambda_bg_reg * loss_bg_reg
        loss = loss_object + loss_bg + lambda_sem * loss_sem

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                iteration=iteration,
                Ll1=Ll1,
                loss=loss,
                l1_loss_fn=l1_loss,
                elapsed=iter_start.elapsed_time(iter_end),
                testing_iterations=testing_iterations,
                scene=scene,
                renderFunc=render,
                renderArgs=(pipe, background),
                loss_sem=loss_sem,
                loss_obj_fg=loss_obj_fg,
                loss_obj_alpha=loss_obj_alpha,
                loss_bg_rgb=loss_bg_rgb,
                loss_bg_reg=loss_bg_reg,
                use_wandb=use_wandb,
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(
                    classifier.state_dict(),
                    os.path.join(
                        scene.model_path,
                        "point_cloud/iteration_{}".format(iteration),
                        "classifier.pth",
                    ),
                )

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                cls_optimizer.step()
                cls_optimizer.zero_grad()

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(
    iteration,
    Ll1,
    loss,
    l1_loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    loss_sem,
    loss_obj_fg,
    loss_obj_alpha,
    loss_bg_rgb,
    loss_bg_reg,
    use_wandb,
):
    if use_wandb:
        log_dict = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_loss_patches/loss_sem": loss_sem.item(),
            "train_loss_patches/loss_obj_fg": loss_obj_fg.item(),
            "train_loss_patches/loss_obj_alpha": loss_obj_alpha.item(),
            "train_loss_patches/loss_bg_rgb": loss_bg_rgb.item(),
            "train_loss_patches/loss_bg_reg": loss_bg_reg.item(),
            "iter_time": elapsed,
            "iter": iteration,
        }
        wandb.log(log_dict)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if use_wandb:
                        if idx < 5:
                            wandb.log(
                                {
                                    config["name"] + "_view_{}/render".format(viewpoint.image_name): [
                                        wandb.Image(image)
                                    ]
                                }
                            )
                            if iteration == testing_iterations[0]:
                                wandb.log(
                                    {
                                        config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name): [
                                            wandb.Image(gt_image)
                                        ]
                                    }
                                )

                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if use_wandb:
                    wandb.log(
                        {
                            config["name"] + "/loss_viewpoint - l1_loss": l1_test,
                            config["name"] + "/loss_viewpoint - psnr": psnr_test,
                        }
                    )

        if use_wandb:
            wandb.log(
                {
                    "scene/opacity_histogram": scene.gaussians.get_opacity,
                    "total_points": scene.gaussians.get_xyz.shape[0],
                    "iter": iteration,
                }
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.iterations = 10000
    args.save_iterations.append(args.iterations)

    try:
        with open(args.config_file, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", DEFAULT_SEMANTIC_CLASSES)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.dgsrsim_lambda_alpha = config.get("dgsrsim_lambda_alpha", config.get("lambda_alpha", 0.10))
    args.dgsrsim_lambda_bg_reg = config.get("dgsrsim_lambda_bg_reg", config.get("lambda_bg_reg", 0.50))
    args.dgsrsim_lambda_sem = config.get("dgsrsim_lambda_sem", config.get("lambda_sem", 1.00))
    args.dgsrsim_background_label = config.get("dgsrsim_background_label", 0)
    args.dgsrsim_knn_k = config.get("dgsrsim_knn_k", 8)
    args.dgsrsim_reg_max_points = config.get("dgsrsim_reg_max_points", 2048)

    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.use_wandb,
    )

    print("\nTraining complete.")
