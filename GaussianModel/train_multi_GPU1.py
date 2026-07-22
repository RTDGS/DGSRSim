raise SystemExit("Legacy training snapshot disabled. Use GaussianModel/train.py.")

# -----------------------------------------------------------------------------------
# FINAL MULTI-GPU VERSION (GPU0: render, GPU1: classifier)
# Fully working version without .to() on GaussianModel
# -----------------------------------------------------------------------------------

import os
import torch
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

# GPU layout
RENDER_DEVICE = "cuda:0"
CLS_DEVICE = "cuda:1"


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, use_wandb):

    first_iter = 0
    prepare_output_and_logger(dataset)

    # ❗ GaussianModel MUST stay on cuda:0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    num_classes = dataset.num_classes
    print("Num classes:", num_classes)

    classifier = None
    cls_optimizer = None

    # Load checkpoint for GaussianModel only
    if checkpoint:
        params, first_iter = torch.load(checkpoint)
        gaussians.restore(params, opt)

    # background (on GPU0)
    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, device=RENDER_DEVICE, dtype=torch.float32)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1

    print("\n✔ Waiting for first render to initialize classifier…\n")

    for iteration in range(first_iter, opt.iterations + 1):

        # GUI connect
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                custom_cam, do_training, pipe.convert_SHs_python, \
                pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()

                net_img_bytes = None
                if custom_cam:
                    out = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    out = torch.clamp(out, 0, 1) * 255
                    net_img_bytes = memoryview(out.byte().permute(1,2,0).cpu().numpy())

                network_gui.send(net_img_bytes, dataset.source_path)
                if do_training:
                    break
            except:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if iteration - 1 == debug_from:
            pipe.debug = True

        # ------------------------- RENDER (GPU0) -------------------------
        render_out = render(viewpoint, gaussians, pipe, background)
        image = render_out["render"]
        objects = render_out["render_object"]
        view_pts = render_out["viewspace_points"]
        visibility_filter = render_out["visibility_filter"]
        radii = render_out["radii"]

        # -------------------- INITIALIZE classifier -----------------------
        if classifier is None:
            in_ch = objects.shape[0]
            print(f"\n✔ Initializing classifier with channels = {in_ch}\n")

            classifier = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1).to(CLS_DEVICE)
            cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

        # -------------------- CLASSIFIER forward on GPU1 --------------------
        objects_4d = objects.unsqueeze(0).to(CLS_DEVICE)
        logits = classifier(objects_4d)
        logits = logits.squeeze(0).to(RENDER_DEVICE)

        gt_obj = viewpoint.objects.to(RENDER_DEVICE).long()
        loss_obj = torch.nn.CrossEntropyLoss(reduction='none')(
            logits.unsqueeze(0), gt_obj.unsqueeze(0)
        ).mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes, device=RENDER_DEVICE))

        # -------------------- IMAGE loss --------------------
        gt_img = viewpoint.original_image.to(RENDER_DEVICE)
        Ll1 = l1_loss(image, gt_img)

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            dc = gaussians._objects_dc.permute(2,0,1).unsqueeze(0).to(CLS_DEVICE)
            logits3d = classifier(dc).squeeze(0)
            prob3d = torch.softmax(logits3d, dim=0).to(RENDER_DEVICE)

            loss_cls_3d_xyz = gaussians._xyz.squeeze().detach().to(CLS_DEVICE)

            loss_obj_3d = loss_cls_3d(
                loss_cls_3d_xyz,
                prob3d,  # both on GPU1
                opt.reg3d_k,
                opt.reg3d_lambda_val,
                opt.reg3d_max_points,
                opt.reg3d_sample_size
            )

            loss = (1-opt.lambda_dssim)*Ll1 + \
                   opt.lambda_dssim*(1-ssim(image, gt_img)) + \
                   loss_obj + loss_obj_3d
        else:
            loss = (1-opt.lambda_dssim)*Ll1 + \
                   opt.lambda_dssim*(1-ssim(image, gt_img)) + \
                   loss_obj

        loss.backward()
        iter_end.record()

        # -------------------- LOGGING & SAVE --------------------
        with torch.no_grad():
            ema_loss = 0.4*loss.item() + 0.6*ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)

            # Save Gaussian + classifier
            if iteration in saving_iterations:
                print(f"\n💾 Saving at iter {iteration}")
                scene.save(iteration)

                save_dir = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pth"))

            # DENSIFY
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(view_pts, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_th = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_th
                    )

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # STEP
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                cls_optimizer.step()
                cls_optimizer.zero_grad()

            # CHECKPOINT
            if iteration in checkpoint_iterations:
                ckpt_path = os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                torch.save((gaussians.capture(), iteration), ckpt_path)


def prepare_output_and_logger(args):
    if not args.model_path:
        uid = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", uid[:10])

    os.makedirs(args.model_path, exist_ok=True)
    print("Output folder:", args.model_path)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def training_report(*args, **kwargs):
    pass


# ------------------------------ MAIN ------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1000,7000,30000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000,7000,30000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", default=None)
    parser.add_argument("--config_file", default="config.json")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Load config JSON
    with open(args.config_file) as f:
        cfg = json.load(f)

    args.densify_until_iter = cfg.get("densify_until_iter", 15000)
    args.num_classes = cfg.get("num_classes", 200)
    args.reg3d_interval = cfg.get("reg3d_interval", 2)
    args.reg3d_k = cfg.get("reg3d_k", 5)
    args.reg3d_lambda_val = cfg.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = cfg.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = cfg.get("reg3d_sample_size", 1000)

    print("Optimizing", args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from, args.use_wandb
    )

    print("\nTraining complete!")
