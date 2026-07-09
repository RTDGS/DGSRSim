import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.ext.automatic_sam import get_sam_model
from deva.ext.automatic_processor import process_frame_automatic as process_frame

from tqdm import tqdm
import json


def apply_memory_saving_patch(cfg: dict) -> dict:
    """
    显存控制（关键）
    注意：必须在 get_sam_model(cfg, ...) 之前调用，否则 SAM 的 AutomaticMaskGenerator 已经初始化，
    后续再改 cfg 不会生效。
    """
    # 1) 降低 SAM AutomaticMaskGenerator 的采样密度与批大小（最有效）
    # 原默认一般是 64/64，非常容易在 ScanNet 爆显存
    cfg['SAM_NUM_POINTS_PER_SIDE'] = 24       # 推荐保命档：24（质量更高可改 32）
    cfg['SAM_NUM_POINTS_PER_BATCH'] = 8       # 推荐保命档：8（可改 16）

    # 2) 限制最多对象数（减少后处理/统计张量堆积）
    cfg['max_num_objects'] = min(cfg.get('max_num_objects', 200), 80)

    # 3) 关闭 long-term（ScanNet 自动伪标签阶段通常不刚需，且更省显存）
    cfg['enable_long_term'] = False
    cfg['enable_long_term_count_usage'] = False

    # 你也可以根据需要进一步收紧（可选）
    # cfg['max_long_term_elements'] = min(cfg.get('max_long_term_elements', 10000), 3000)

    return cfg


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)

    """
    Arguments loading
    """
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)

    deva_model, cfg, args = get_model_and_config(parser)

    """
    关键：先改 cfg，再初始化 SAM
    """
    cfg = apply_memory_saving_patch(cfg)

    # 初始化 SAM（此时会读取 cfg['SAM_NUM_POINTS_PER_SIDE/BATCH'] 等）
    sam_model = get_sam_model(cfg, 'cuda')

    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    # Start eval
    vid_length = len(loader)

    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg.get('enable_long_term', False)
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements']
    )

    # 打印最终配置（这里应当能看到 24/8 或你设置的值，而不是 64/64）
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
            process_frame(deva, sam_model, im_path, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)

    result_saver.end()

    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json
