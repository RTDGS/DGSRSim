# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             use_wandb):
    # --- DDP Setup ---
    # 获取本地 Rank，并立即设置 GPU 设备，防止后续 Tensor 默认去 cuda:0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_master = (local_rank == 0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)  # 再次强制设置

    first_iter = 0
    if is_master:
        prepare_output_and_logger(dataset)

    dist.barrier()  # 等待主进程创建目录

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    num_classes = dataset.num_classes
    if is_master:
        print("Num classes: ", num_classes)

    # --- Classifier DDP ---
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.to(device)
    classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, map_location=device)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=not is_master)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if is_master and network_gui.conn == None:
            network_gui.try_connect()
        while is_master and network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --- Data Sharding ---
        if not viewpoint_stack:
            all_cameras = scene.getTrainCameras().copy()
            # 简单的取模分片
            viewpoint_stack = [cam for i, cam in enumerate(all_cameras) if i % world_size == local_rank]

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # --- CRITICAL FIX: Ensure Camera Tensors are on Current Device ---
        # 很多时候 Camera 初始化时 tensors 被放到了 cuda:0 或者其他地方
        # 这里强制将用于 Rasterizer 的矩阵移动到当前 device
        if viewpoint_cam.world_view_transform.device != device:
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to(device)
        if viewpoint_cam.projection_matrix.device != device:
            viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.to(device)
        if viewpoint_cam.full_proj_transform.device != device:
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.to(device)
        if viewpoint_cam.camera_center.device != device:
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.to(device)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 此时 viewpoint_cam 的内部 tensor 已经保证在 device 上了
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        # Object Loss
        gt_obj = viewpoint_cam.objects.to(device).long()
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))

        # Loss
        gt_image = viewpoint_cam.original_image.to(device)
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            # 确保 gaussians 参数也在 device 上 (通常是，但为了保险)
            logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
            prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1, 0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val,
                                      opt.reg3d_max_points, opt.reg3d_sample_size)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                        1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj

        loss.backward()

        # --- Manual Gradient Synchronization ---
        for group in gaussians.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                if is_master:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
            if iteration == opt.iterations:
                if is_master:
                    progress_bar.close()

            # Log and save
            if is_master:
                training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations,
                                scene, render, (pipe, background), loss_obj_3d, use_wandb)

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    torch.save(classifier.state_dict(),
                               os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),
                                            'classifier.pth'))

            # --- Distributed Densification ---
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    if is_master:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    dist.barrier()

                    # Broadcast Params
                    model_state_container = [None]
                    if is_master:
                        model_state_container = [gaussians.capture()]

                    dist.broadcast_object_list(model_state_container, src=0)

                    if not is_master:
                        gaussians.restore(model_state_container[0], opt)

                    gaussians.optimizer.zero_grad(set_to_none=True)
                    cls_optimizer.zero_grad()

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    if is_master:
                        gaussians.reset_opacity()
                    dist.barrier()
                    if not is_master:
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

            if (iteration in checkpoint_iterations) and is_master:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs,
                    loss_obj_3d, use_wandb):
    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(),
                       "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(),
                       "iter_time": elapsed, "iter": iteration})

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Render 需要保证 tensor 在正确设备，但在 test/report 阶段我们通常只在 master 运行
                    # 这里为了简化，假设 master 的设备是 cuda:0，而 scene.gaussians 已经在 cuda:0
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log(
                                {config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [
                                    wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test,
                               config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity,
                       "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    # --- DDP Init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # 最早时间点设置
    dist.init_process_group(backend="nccl")

    if local_rank == 0:
        print("Optimizing " + args.model_path)
        if args.use_wandb:
            wandb.init(project="gaussian-splatting")
            wandb.config.args = args
            wandb.run.name = args.model_path

    safe_state(args.quiet)
    torch.manual_seed(42 + local_rank)

    if local_rank == 0:
        network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    dist.destroy_process_group()
    if local_rank == 0:
        print("\nTraining complete.")