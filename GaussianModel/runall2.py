raise SystemExit("Legacy orchestration snapshot disabled. Follow README.md canonical steps.")

import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# HARD-CODED SETTINGS (edit here only)
# ============================================================

# 1) Hard-code multiple dataset folders under ./data/
DATASETS = [
    # "yumi",
    # "hu",
    # "guo",
    # "toumingguogai",
    # "pingdiguo",
    # "gaoyaguo",
    # "gaoyaguogai",
    # "guochan",
    # "beizi",
    # "diancilu",
    # "wan",
"background",

]

# 2) Hard-code run id
RUN_ID = "1"

# 3) Hard-code prompt (for DEVA demo_with_text.py)
PROMPT = "black blurry hole"

# 4) Gray ID strategy:
#    - If REUSE_GRAY_ID is set to an integer 0..255, the script will NOT open Gray1.py,
#      and will use this value for ALL datasets.
#    - If REUSE_GRAY_ID is None, the script will open Gray1.py for EACH dataset and you click once.
REUSE_GRAY_ID = None   # e.g. 32, or None


# ============================================================
# Helper functions
# ============================================================

def run(cmd: str, cwd: Optional[Path] = None, env: Optional[Dict] = None) -> None:
    print("\n" + "=" * 100)
    print("[CMD] {0}".format(cmd))
    print("[CWD] {0}".format(str(cwd) if cwd else str(PROJECT_ROOT)))
    print("=" * 100)
    subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def update_select_obj_id(config_path: Path, ids: List[int]) -> None:
    if not config_path.exists():
        raise FileNotFoundError("Config not found: {0}".format(config_path))

    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_bytes(config_path.read_bytes())
        print("[INFO] Backup created: {0}".format(backup_path))

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["select_obj_id"] = ids

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("[INFO] Updated select_obj_id={0} in {1}".format(ids, config_path))


def read_picked_gray(pick_json: Path) -> int:
    if not pick_json.exists():
        raise FileNotFoundError("Pick result not found: {0}".format(pick_json))
    obj = json.loads(pick_json.read_text(encoding="utf-8"))
    v = int(obj.get("target_gray"))
    if v < 0 or v > 255:
        raise ValueError("Picked gray is out of range [0, 255]")
    return v


def wait_for_object_mask_image(object_mask_dir: Path, timeout_sec: int = 1800, poll_sec: int = 5) -> Path:
    start = time.time()
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    while True:
        if object_mask_dir.exists():
            candidates = sorted(
                p for p in object_mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
            if candidates:
                return candidates[0]

        elapsed = int(time.time() - start)
        if elapsed >= timeout_sec:
            raise TimeoutError(
                "Timed out waiting for object_mask images. Dir: {0}, waited: {1}s"
                .format(object_mask_dir, elapsed)
            )

        print("[WAIT] object_mask not ready, sleeping {0}s ... (elapsed {1}s)".format(poll_sec, elapsed))
        time.sleep(poll_sec)


def pick_gray_via_gui(dataset: str, gray_script: Path, gray_image_path: Path) -> int:
    """
    Calls Gray1.py which writes {"target_gray": <int>} into a json.
    """
    pick_json = PROJECT_ROOT / (".picked_gray.{0}.json".format(dataset))
    if pick_json.exists():
        try:
            pick_json.unlink()
        except Exception:
            pass

    run("{0} {1} {2} {3}".format(sys.executable, gray_script, gray_image_path, pick_json), cwd=PROJECT_ROOT)
    return read_picked_gray(pick_json)


def run_one_dataset(dataset: str, target_gray_override: Optional[int]) -> None:
    print("\n" + "#" * 100)
    print("[DATASET] {0}".format(dataset))
    print("#" * 100)

    gray_script = PROJECT_ROOT / "Gray1.py"
    change_gray_script = PROJECT_ROOT / "changeGray1.py"

    if not change_gray_script.exists():
        raise FileNotFoundError("changeGray1.py not found: {0}".format(change_gray_script))

    # Config paths
    removal_config = PROJECT_ROOT / "config/object_removal/astronaut.json"
    inpaint_config = PROJECT_ROOT / "config/object_inpaint/astronaut.json"

    # -------------------------
    # Step 0: convert / prepare_pseudo_label / train
    # -------------------------
    run("python convert.py -s data/{0}".format(dataset), cwd=PROJECT_ROOT)
    run("bash script/prepare_pseudo_label_1.sh {0} {1}".format(dataset, RUN_ID), cwd=PROJECT_ROOT)
    run("bash script/train.sh {0} {1}".format(dataset, RUN_ID), cwd=PROJECT_ROOT)

    # -------------------------
    # Step A: object_mask generated by train.sh -> wait -> pick gray (or reuse hard-coded)
    # -------------------------
    if target_gray_override is not None:
        target_gray = int(target_gray_override)
        if target_gray < 0 or target_gray > 255:
            raise ValueError("REUSE_GRAY_ID must be in [0, 255]")
        print("[INFO] Using hard-coded target_gray = {0}".format(target_gray))
    else:
        # need GUI pick
        if not (PROJECT_ROOT / "Gray1.py").exists():
            raise FileNotFoundError("Gray1.py not found: {0}".format(PROJECT_ROOT / "Gray1.py"))

        object_mask_dir = PROJECT_ROOT / ("data/{0}/object_mask".format(dataset))
        gray_image_path = wait_for_object_mask_image(object_mask_dir, timeout_sec=1800, poll_sec=5)
        print("[INFO] Using gray image for picking: {0}".format(gray_image_path))

        target_gray = pick_gray_via_gui(dataset, gray_script, gray_image_path)
        print("[INFO] Picked target_gray = {0}".format(target_gray))

    # Update both configs
    update_select_obj_id(removal_config, [target_gray])
    update_select_obj_id(inpaint_config, [target_gray])

    # -------------------------
    # Step 1: object_removal
    # -------------------------
    run(
        "bash script/edit_object_removal.sh output/{0} config/object_removal/astronaut.json --skip_test".format(dataset),
        cwd=PROJECT_ROOT,
    )

    # input("[MANUAL] If you need reverse selection, modify gaussian model line ~218 now. Press Enter to continue...")
    #
    # # -------------------------
    # # Step 2-1: DEVA (auto-create directories)
    # # -------------------------
    # deva_dir = PROJECT_ROOT / "Tracking-Anything-with-DEVA"
    # lama_dir = PROJECT_ROOT / "lama"
    #
    # mask_dir = deva_dir / ("output_2d_inpaint_mask/{0}".format(dataset))
    # lama_test_dir = lama_dir / ("LaMa_test_images/{0}".format(dataset))
    # mask_dir.mkdir(parents=True, exist_ok=True)
    # lama_test_dir.mkdir(parents=True, exist_ok=True)
    #
    # renders_dir = PROJECT_ROOT / ("output/{0}/train/ours_object_removal/iteration_30000/renders".format(dataset))
    # if not renders_dir.exists():
    #     raise FileNotFoundError("Renders folder not found: {0}".format(renders_dir))
    #
    # run(
    #     "python demo/demo_with_text.py "
    #     "--chunk_size 4 "
    #     "--img_path ../output/{0}/train/ours_object_removal/iteration_30000/renders "
    #     "--amp "
    #     "--temporal_setting semionline "
    #     "--size 480 "
    #     "--output ./output_2d_inpaint_mask/{0} "
    #     "--prompt \"{1}\"".format(dataset, PROMPT),
    #     cwd=deva_dir,
    # )
    #
    # run(
    #     "python prepare_lama_input.py "
    #     "../output/{0}/train/ours_object_removal/iteration_30000/renders "
    #     "./output_2d_inpaint_mask/{0} "
    #     "../lama/LaMa_test_images/{0}".format(dataset),
    #     cwd=deva_dir,
    # )
    #
    # # -------------------------
    # # Step 2-2: LaMa predict + prepare_pseudo_label
    # # -------------------------
    # lama_env = os.environ.copy()
    # lama_env["TORCH_HOME"] = str(lama_dir)
    # lama_env["PYTHONPATH"] = str(lama_dir)
    #
    # run(
    #     "python bin/predict.py "
    #     "refine=True "
    #     "model.path={0}/big-lama "
    #     "indir={0}/LaMa_test_images/{1} "
    #     "outdir={0}/output/{1}".format(str(lama_dir), dataset),
    #     cwd=lama_dir,
    #     env=lama_env,
    # )
    #
    # run(
    #     "python prepare_pseudo_label.py {0}/output/{1} {2}/data/{1}".format(str(lama_dir), dataset, str(PROJECT_ROOT)),
    #     cwd=lama_dir,
    #     env=lama_env,
    # )
    #
    # # -------------------------
    # # Step 2.5: changeGray BEFORE Step3
    # # -------------------------
    # input_folder = PROJECT_ROOT / ("data/{0}/object_mask".format(dataset))
    # output_folder = PROJECT_ROOT / ("data/{0}/inpaint_object_mask_255".format(dataset))
    #
    # run(
    #     "{0} {1} {2} {3} {4}".format(sys.executable, change_gray_script, input_folder, output_folder, target_gray),
    #     cwd=PROJECT_ROOT,
    # )
    #
    # # -------------------------
    # # Step 3: object_inpaint
    # # -------------------------
    # run(
    #     "bash script/edit_object_inpaint.sh output/{0} config/object_inpaint/astronaut.json".format(dataset),
    #     cwd=PROJECT_ROOT,
    # )

    print("\n[SUCCESS] Dataset completed: {0}".format(dataset))


def main() -> int:
    print("[INFO] PROJECT_ROOT = {0}".format(PROJECT_ROOT))
    print("[INFO] RUN_ID       = {0}".format(RUN_ID))
    print("[INFO] PROMPT       = {0}".format(PROMPT))
    print("[INFO] DATASETS     = {0}".format(", ".join(DATASETS)))
    print("[INFO] REUSE_GRAY_ID= {0}".format(REUSE_GRAY_ID))

    for ds in DATASETS:
        try:
            run_one_dataset(ds, REUSE_GRAY_ID)
        except Exception as e:
            print("\n[ERROR] Dataset failed: {0}".format(ds))
            print("[ERROR] {0}".format(e))
            # continue with next dataset
            continue

    print("\n[ALL DONE] Batch processing finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
