import os
from os import path
from argparse import ArgumentParser

# Must be set before CUDA is initialized. The shell script also sets this.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.ext.automatic_sam import get_sam_model
import deva.ext.automatic_processor as automatic_processor

from tqdm import tqdm
import json


def _as_torch_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available, but a CUDA device was requested.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f'Requested {device_str}, but only {torch.cuda.device_count()} CUDA device(s) are visible. '
                'Check CUDA_VISIBLE_DEVICES or pass different --deva_device/--sam_device values.'
            )
    return device


def _move_tensors(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_move_tensors(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_tensors(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def _target_hw_from_min_side(h: int, w: int, min_side: int):
    if min_side is None or min_side <= 0:
        return h, w
    scale = float(min_side) / float(min(h, w))
    return int(h * scale), int(w * scale)


def _resize_image_for_sam(image_np: np.ndarray, min_side: int, sam_max_side: int):
    """
    Downsample only the image sent into SAM, not the frame stored/saved by DEVA.

    The resize dimensions are chosen as an integer multiple of DEVA's internal target
    resolution, so auto_segment(..., min_side=cfg['size']) still returns a mask with
    the same HxW expected by DEVA.
    """
    if sam_max_side is None or sam_max_side <= 0:
        return image_np, None

    h, w = image_np.shape[:2]
    target_h, target_w = _target_hw_from_min_side(h, w, min_side)
    target_long = max(target_h, target_w)
    if target_long <= 0:
        return image_np, None

    multiplier = max(1, int(sam_max_side) // target_long)
    sam_h = target_h * multiplier
    sam_w = target_w * multiplier

    # Never upscale the input just for SAM. Upscaling costs memory and adds no detail.
    if sam_h >= h and sam_w >= w:
        return image_np, None

    resized = cv2.resize(image_np, (sam_w, sam_h), interpolation=cv2.INTER_AREA)
    info = {
        'original_hw': (h, w),
        'sam_hw': (sam_h, sam_w),
        'deva_target_hw': (target_h, target_w),
        'multiplier': multiplier,
    }
    return resized, info


def _resize_index_mask(mask: torch.Tensor, target_hw):
    if not torch.is_tensor(mask) or mask.ndim != 2:
        return mask
    if tuple(mask.shape[-2:]) == tuple(target_hw):
        return mask
    resized = F.interpolate(
        mask[None, None].float(),
        size=target_hw,
        mode='nearest',
    )[0, 0]
    return resized.to(dtype=mask.dtype, device=mask.device)


def _ensure_first_mask_shape(result, target_hw):
    """Resize only the first returned tensor mask if rounding made it off by a pixel."""
    if target_hw is None:
        return result
    if torch.is_tensor(result):
        return _resize_index_mask(result, target_hw)
    if isinstance(result, tuple) and len(result) > 0:
        first = _resize_index_mask(result[0], target_hw)
        return (first,) + result[1:]
    if isinstance(result, list) and len(result) > 0:
        out = list(result)
        out[0] = _resize_index_mask(out[0], target_hw)
        return out
    return result


def _is_cuda_oom(err: BaseException) -> bool:
    text = str(err).lower()
    return ('cuda out of memory' in text) or ('outofmemoryerror' in text) or ('out of memory' in text and 'cuda' in text)


def _empty_cuda_cache(*devices):
    if not torch.cuda.is_available():
        return
    seen = set()
    for device in devices:
        device = torch.device(device)
        if device.type != 'cuda':
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        if index in seen:
            continue
        seen.add(index)
        try:
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        except Exception:
            # Cache clearing is best effort; do not hide the original error.
            pass


def _install_two_gpu_oomsafe_bridge(deva_device: torch.device):
    """
    DEVA runs on deva_device, SAM can run on another CUDA device.
    This bridge also reduces SAM peak memory by optionally downsampling the SAM input
    and retrying with smaller SAM batches after CUDA OOM.
    """
    original_make_segmentation = automatic_processor.make_segmentation

    def make_segmentation_device_bridge(cfg, image_np, forward_mask, sam_model, *args, **kwargs):
        predictor = getattr(sam_model, 'predictor', None)
        sam_device = getattr(predictor, 'device', None)
        if sam_device is None:
            sam_device = torch.device(cfg.get('sam_device', 'cuda:1'))
        else:
            sam_device = torch.device(sam_device)

        if forward_mask is not None:
            forward_mask = forward_mask.to(device=sam_device, non_blocking=True)

        min_side = int(args[0]) if len(args) >= 1 and args[0] is not None else int(cfg.get('size', 480))
        target_hw = _target_hw_from_min_side(image_np.shape[0], image_np.shape[1], min_side)

        base_max_side = int(cfg.get('sam_max_side', 1280) or 0)
        oom_retries = int(cfg.get('sam_oom_retries', 2) or 0)
        min_points_per_batch = int(cfg.get('sam_min_points_per_batch', 1) or 1)

        last_err_msg = None
        for attempt in range(oom_retries + 1):
            if base_max_side > 0:
                # Attempt 0 uses base_max_side; later attempts shrink the SAM input.
                current_max_side = max(640, int(base_max_side * (0.75 ** attempt)))
            else:
                current_max_side = 0

            image_for_sam, resize_info = _resize_image_for_sam(image_np, min_side, current_max_side)
            if resize_info is not None and attempt == 0:
                print(
                    '[OOMSAFE] SAM input resized '
                    f'{resize_info["original_hw"]} -> {resize_info["sam_hw"]}; '
                    f'DEVA mask target={resize_info["deva_target_hw"]}'
                )

            try:
                _empty_cuda_cache(sam_device)
                with torch.cuda.device(sam_device) if sam_device.type == 'cuda' else torch.no_grad():
                    result = original_make_segmentation(
                        cfg, image_for_sam, forward_mask, sam_model, *args, **kwargs
                    )
                result = _move_tensors(result, deva_device)
                result = _ensure_first_mask_shape(result, target_hw)
                return result
            except RuntimeError as err:
                if not _is_cuda_oom(err):
                    raise
                last_err_msg = str(err)
                old_batch = int(getattr(sam_model, 'points_per_batch', cfg.get('SAM_NUM_POINTS_PER_BATCH', 64)))
                new_batch = max(min_points_per_batch, old_batch // 2)
                if new_batch < old_batch:
                    sam_model.points_per_batch = new_batch
                print(
                    '[OOMSAFE] Caught CUDA OOM inside SAM. '
                    f'attempt={attempt + 1}/{oom_retries + 1}, '
                    f'SAM_NUM_POINTS_PER_BATCH {old_batch}->{getattr(sam_model, "points_per_batch", old_batch)}, '
                    f'sam_max_side={current_max_side}. Retrying...'
                )
                _empty_cuda_cache(deva_device, sam_device)

        if last_err_msg is not None:
            raise RuntimeError('SAM segmentation still ran out of CUDA memory after OOM-safe retries. '
                               f'Last CUDA OOM: {last_err_msg}')
        raise RuntimeError('SAM segmentation failed without a captured exception.')

    automatic_processor.make_segmentation = make_segmentation_device_bridge


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)

    parser = ArgumentParser()

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)

    # Two-GPU placement. DEVA must be the default CUDA device because upstream
    # code uses .cuda() in get_model_and_config() and get_input_frame_for_deva().
    parser.add_argument('--deva_device', default='cuda:0', help='Device for DEVA propagation/tracking.')
    parser.add_argument('--sam_device', default='cuda:1', help='Device for SAM automatic mask generation.')
    parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers.')
    parser.add_argument('--empty_cache_every', default=1, type=int, help='Call torch.cuda.empty_cache() every N frames; 0 disables it.')

    # OOM controls for automatic SAM.
    parser.add_argument(
        '--sam_max_side',
        default=1280,
        type=int,
        help='Downsample only the image sent to SAM so its long side is at most this value. 0 disables.',
    )
    parser.add_argument(
        '--sam_oom_retries',
        default=2,
        type=int,
        help='Retry SAM after CUDA OOM by reducing SAM batch size and SAM input size.',
    )
    parser.add_argument(
        '--sam_min_points_per_batch',
        default=1,
        type=int,
        help='Lower bound for automatic retry reduction of SAM_NUM_POINTS_PER_BATCH.',
    )

    # Parse once before get_model_and_config() so .cuda() goes to deva_device.
    preliminary_args, _ = parser.parse_known_args()
    deva_device = _as_torch_device(preliminary_args.deva_device)
    sam_device = _as_torch_device(preliminary_args.sam_device)

    if deva_device.type == 'cuda':
        torch.cuda.set_device(deva_device)

    deva_model, cfg, args = get_model_and_config(parser)

    # Keep explicit custom values in cfg for the bridge and logging/debugging.
    cfg['deva_device'] = args.deva_device
    cfg['sam_device'] = args.sam_device
    cfg['sam_max_side'] = args.sam_max_side
    cfg['sam_oom_retries'] = args.sam_oom_retries
    cfg['sam_min_points_per_batch'] = args.sam_min_points_per_batch

    deva_device = _as_torch_device(args.deva_device)
    sam_device = _as_torch_device(args.sam_device)
    if deva_device.type == 'cuda':
        torch.cuda.set_device(deva_device)
    deva_model = deva_model.to(device=deva_device).eval()

    _install_two_gpu_oomsafe_bridge(deva_device)
    sam_model = get_sam_model(cfg, str(sam_device))

    print(f'Using devices: DEVA={deva_device}, SAM={sam_device}')
    print(
        'SAM OOM controls: '
        f'SAM_NUM_POINTS_PER_SIDE={cfg.get("SAM_NUM_POINTS_PER_SIDE")}, '
        f'SAM_NUM_POINTS_PER_BATCH={cfg.get("SAM_NUM_POINTS_PER_BATCH")}, '
        f'SAM_PRED_IOU_THRESHOLD={cfg.get("SAM_PRED_IOU_THRESHOLD")}, '
        f'sam_max_side={cfg.get("sam_max_side")}'
    )

    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=args.num_workers)
    out_path = cfg['output']

    vid_length = len(loader)
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = args.num_voting_frames - 1
    if args.use_short_id:
        pass
    else:
        deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    with torch.cuda.amp.autocast(enabled=args.amp):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame = automatic_processor.process_frame_automatic
            process_frame(deva, sam_model, im_path, result_saver, ti, image_np=frame)
            if args.empty_cache_every > 0 and (ti + 1) % args.empty_cache_every == 0:
                _empty_cuda_cache(deva_device, sam_device)
        flush_buffer(deva, result_saver)
    result_saver.end()

    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)
