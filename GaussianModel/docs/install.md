# Legacy upstream installation note

This page preserves the installation recipe inherited from Gaussian Grouping. It is not the canonical DGSRSim paper environment and its PyTorch 1.12 stack should not be used to interpret the reported experiments. Use [`../../requirements/README.md`](../../requirements/README.md) and [`../../requirements/paper_environment.md`](../../requirements/paper_environment.md) for the DGSRSim environment record. DEVA below is optional mask preparation tooling; `train.py` consumes prepared masks and does not invoke DEVA.

### Upstream installation


Clone the repository locally
```
git clone https://github.com/lkeab/gaussian-grouping.git
cd gaussian-grouping
```

Our default, provided install method is based on Conda package and environment management:
```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

(Optional) If you want to prepare masks on your own dataset, you will also need to prepare [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) environment.

```bash
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

(Optional) If you want to inpaint on your own dataset, you will also need to prepare [LaMa](https://github.com/advimman/lama) environment.

```bash
cd lama
pip install -r requirements.txt
cd ..
```
