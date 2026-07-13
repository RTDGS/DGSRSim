# -*- coding: utf-8 -*-
"""
rgb_segment_click_select_service.py

What you asked for
------------------
Refactor the offline click-select segmentation logic into a reusable module with:

1) Model lifecycle:
   - load_fastsam_model(...)     : load ONCE
   - unload_fastsam_model()      : free resources
2) Inference:
   - infer_instance_masks(...)   : run segmentation using the loaded model (NO reload)
3) UI click-select:
   - ui_click_select_mask(...)   : show [RGB | INST], click selects immediately
   - optional save (press S), reset (R), quit (Q/ESC)

Key point:
- The model is cached in a global service object so repeated inference does not reload.

Dependencies
------------
pip install numpy opencv-python
FastSAM must be importable in current env:
  from fastsam import FastSAM, FastSAMPrompt

Usage
-----
from rgb_segment_click_select_service import (
    load_fastsam_model, unload_fastsam_model,
    infer_instance_masks, ui_click_select_mask
)

load_fastsam_model(model_path=".../FastSAM-x.pt", device="cuda")
masks = infer_instance_masks(color_bgr, img_size=1024, conf=0.4, iou=0.9, min_area=80)
sel = ui_click_select_mask(color_bgr, masks, allow_save=True, out_dir="...", stem="000001")
unload_fastsam_model()
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2


# =========================
# IO helpers
# =========================

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_mask_png(mask_bool: np.ndarray, out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    m = (mask_bool.astype(np.uint8) * 255)
    ok = cv2.imwrite(out_png, m)
    if not ok:
        raise RuntimeError(f"Failed to write mask png: {out_png}")


# =========================
# FastSAM model service
# =========================

@dataclass
class FastSAMServiceConfig:
    model_path: str
    device: str = "cuda"


class FastSAMService:
    """
    Holds a loaded FastSAM model to avoid reloading each inference.
    """
    def __init__(self, cfg: FastSAMServiceConfig):
        self.cfg = cfg
        try:
            from fastsam import FastSAM  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FastSAM import failed. Ensure you run inside FastSAM repo/env.\n"
                f"Import error: {repr(e)}"
            )
        self._FastSAM = FastSAM
        self.model = self._FastSAM(self.cfg.model_path)

    def infer_masks(
        self,
        color_bgr: np.ndarray,
        img_size: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        min_area: int = 80,
    ) -> np.ndarray:
        """
        Run FastSAM inference using loaded model; returns masks_bool (N,H,W).
        """
        try:
            from fastsam import FastSAMPrompt  # type: ignore
        except Exception as e:
            raise RuntimeError(f"FastSAMPrompt import failed: {repr(e)}")

        img_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        results = self.model(
            img_rgb,
            device=self.cfg.device,
            retina_masks=True,
            imgsz=int(img_size),
            conf=float(conf),
            iou=float(iou),
        )
        prompt = FastSAMPrompt(img_rgb, results, device=self.cfg.device)

        masks = None
        try:
            masks = prompt.everything_prompt()
        except Exception:
            pass
        if masks is None:
            try:
                masks = prompt.segment()
            except Exception as e:
                raise RuntimeError(f"FastSAMPrompt API mismatch; cannot extract masks: {repr(e)}")

        # to numpy
        masks_np = None
        try:
            import torch  # type: ignore
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks)
        except Exception:
            masks_np = np.asarray(masks)

        if masks_np is None or masks_np.ndim != 3:
            raise RuntimeError(f"Unexpected masks format. Expect (N,H,W). Got: {None if masks_np is None else masks_np.shape}")

        masks_bool = masks_np.astype(bool)

        # filter tiny masks
        areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
        keep = areas >= int(min_area)
        masks_bool = masks_bool[keep]
        if masks_bool.shape[0] == 0:
            return masks_bool

        # ensure same H,W as input
        H, W = color_bgr.shape[:2]
        if masks_bool.shape[1] != H or masks_bool.shape[2] != W:
            resized = []
            for i in range(masks_bool.shape[0]):
                m = (masks_bool[i].astype(np.uint8) * 255)
                m2 = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0
                resized.append(m2)
            masks_bool = np.stack(resized, axis=0)

        return masks_bool

    def unload(self) -> None:
        """
        Best-effort resource cleanup.
        """
        self.model = None  # type: ignore
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        # If torch is used and GPU memory matters:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# Global singleton service
_service: Optional[FastSAMService] = None


def load_fastsam_model(model_path: str, device: str = "cuda") -> None:
    """
    Load FastSAM model once and keep it cached globally.
    Calling again overwrites the existing one (unloads old first).
    """
    global _service
    if _service is not None:
        _service.unload()
        _service = None
    _service = FastSAMService(FastSAMServiceConfig(model_path=str(model_path), device=str(device)))


def unload_fastsam_model() -> None:
    """
    Unload cached FastSAM model.
    """
    global _service
    if _service is None:
        return
    try:
        _service.unload()
    finally:
        _service = None


def infer_instance_masks(
    color_bgr: np.ndarray,
    img_size: int = 1024,
    conf: float = 0.4,
    iou: float = 0.9,
    min_area: int = 80,
) -> np.ndarray:
    """
    Run segmentation using the cached model (no reload).
    You MUST call load_fastsam_model(...) first.
    """
    if _service is None:
        raise RuntimeError("FastSAM model not loaded. Call load_fastsam_model(model_path, device) first.")
    return _service.infer_masks(
        color_bgr=color_bgr,
        img_size=img_size,
        conf=conf,
        iou=iou,
        min_area=min_area,
    )


# =========================
# Click-select UI (matches your original behavior)
# =========================

def build_instance_color_image(masks_bool: np.ndarray, seed: int = 12345) -> np.ndarray:
    if masks_bool.ndim != 3:
        raise ValueError("masks_bool must be (N,H,W)")
    N, H, W = masks_bool.shape
    inst = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(int(seed))
    colors = rng.integers(0, 255, size=(N, 3), dtype=np.uint8)
    for i in range(N):
        inst[masks_bool[i]] = colors[i]
    return inst


def overlay_mask(color_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = color_bgr.copy()
    red = np.zeros_like(out, dtype=np.uint8)
    red[:, :, 2] = 255
    m = mask_bool.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - float(alpha)) + red[m].astype(np.float32) * float(alpha)).astype(np.uint8)
    return out


def pick_mask_by_click(masks_bool: np.ndarray, x: int, y: int) -> Optional[int]:
    if masks_bool.shape[0] == 0:
        return None
    hit = masks_bool[:, y, x]
    idxs = np.where(hit)[0]
    if idxs.size == 0:
        return None
    if idxs.size == 1:
        return int(idxs[0])
    areas = masks_bool[idxs].reshape(idxs.size, -1).sum(axis=1)
    return int(idxs[np.argmin(areas)])


def ui_click_select_mask(
    color_bgr: np.ndarray,
    masks_bool: np.ndarray,
    *,
    out_dir: Optional[str] = None,
    stem: Optional[str] = None,
    source_color_path: Optional[str] = None,
    save_name: str = "mask_00.png",
    win: str = "SegMasks: click to select | S save | R reset | Q quit",
) -> Optional[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    UI behavior aligned with your original script:
      - Show [RGB | INST] side-by-side
      - Click selects immediately (smallest-area if multiple)
      - S: save mask png + json
      - R: reset
      - Q/ESC: quit without saving

    Returns:
      None if user quit without saving.
      Else (saved_png_path, selected_mask_bool, info_dict).
    """
    if masks_bool.ndim != 3:
        raise ValueError("masks_bool must be (N,H,W)")
    H, W = color_bgr.shape[:2]
    if masks_bool.shape[1] != H or masks_bool.shape[2] != W:
        raise ValueError("masks_bool resolution must match color_bgr")

    inst_color = build_instance_color_image(masks_bool)
    selected_idx = {"i": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # window shows [RGB | INST] with same width W each => total 2W
        if x < W:
            xx, yy = x, y
        else:
            xx, yy = x - W, y
        if 0 <= xx < W and 0 <= yy < H:
            mi = pick_mask_by_click(masks_bool, int(xx), int(yy))
            selected_idx["i"] = mi
            print(f"[Click] at ({xx},{yy}) -> selected_mask={mi}")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        left = color_bgr.copy()
        right = inst_color.copy()

        mi = selected_idx["i"]
        if mi is not None:
            left = overlay_mask(left, masks_bool[int(mi)], alpha=0.45)
            edge = cv2.Canny((masks_bool[int(mi)].astype(np.uint8) * 255), 80, 160)
            right[edge > 0] = (0, 0, 255)

        show = np.hstack([left, right])
        cv2.putText(
            show,
            f"masks={masks_bool.shape[0]}  selected={mi}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, show)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            return None

        if k == ord("r"):
            selected_idx["i"] = None

        if k == ord("s"):
            if mi is None:
                print("[Warn] No mask selected. Click on the object first.")
                continue
            if out_dir is None:
                raise ValueError("out_dir is required to save. Provide out_dir=... when calling ui_click_select_mask().")

            ensure_dir(out_dir)
            out_png = os.path.join(out_dir, save_name)
            save_mask_png(masks_bool[int(mi)], out_png)

            out_json = os.path.join(out_dir, os.path.splitext(save_name)[0] + ".json")
            info = {
                "frame_id": stem,
                "timestamp_unix": float(time.time()),
                "source_color_path": source_color_path,
                "mask_png": os.path.basename(out_png),
                "mask_resolution": [int(W), int(H)],
                "selected_mask_index": int(mi),
                "selection_policy": "click pixel -> choose smallest-area mask covering pixel",
            }
            save_json(out_json, info)

            print(f"[Saved] {out_png}")
            cv2.destroyAllWindows()
            return out_png, masks_bool[int(mi)].copy(), {**info, "saved_png": out_png, "saved_json": out_json}


# =========================
# Demo: offline single image
# =========================

def _read_color_any(root: str, stem: str) -> Tuple[np.ndarray, str]:
    p_png = os.path.join(root, "color", f"{stem}.png")
    p_jpg = os.path.join(root, "color", f"{stem}.jpg")
    p = p_png if os.path.isfile(p_png) else p_jpg
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing color for {stem}: {p_png} or {p_jpg}")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read color image: {p}")
    return img, p


if __name__ == "__main__":
    # -------- parameters you edit here --------
    _HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.environ.get("DGSRSIM_RGBD_DUMP_DIR", os.path.join(_HERE, "k2_dump"))
    STEM = "000001"

    OUT_DIR = os.path.join(ROOT, "masks", STEM)

    FASTSAM_MODEL = os.environ.get(
        "DGSRSIM_FASTSAM_MODEL",
        os.path.join(_HERE, "weights", "FastSAM-x.pt"),
    )
    DEVICE = "cuda"
    IMGSZ = 1024
    CONF = 0.4
    IOU = 0.9
    MIN_AREA = 80

    # -------- run --------
    color_bgr, color_path = _read_color_any(ROOT, STEM)

    load_fastsam_model(FASTSAM_MODEL, device=DEVICE)
    try:
        masks = infer_instance_masks(
            color_bgr,
            img_size=IMGSZ,
            conf=CONF,
            iou=IOU,
            min_area=MIN_AREA,
        )
        print(f"[Seg] masks={masks.shape[0]} img={color_bgr.shape}")

        if masks.shape[0] == 0:
            print("[Seg] No masks produced.")
            raise SystemExit(0)

        saved = ui_click_select_mask(
            color_bgr,
            masks,
            out_dir=OUT_DIR,
            stem=STEM,
            source_color_path=os.path.relpath(color_path, ROOT).replace("\\", "/"),
            save_name="mask_00.png",
        )
        print("[Done] saved =", None if saved is None else saved[0])

    finally:
        unload_fastsam_model()
