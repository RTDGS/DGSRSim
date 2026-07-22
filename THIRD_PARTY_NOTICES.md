# 第三方来源与引用

本发布版仓库以 DGSRSim 论文代码为主，将原始子项目 README 合并到主 README 的模块说明中。第三方来源、许可证和引用在本文件集中列出。

根目录 `LICENSE` 中的 AGPL-3.0 仅覆盖未附带其他许可证声明的 DGSRSim 自有代码与文档，不覆盖或替代下列第三方条款。`NOTICE.md` 给出整个混合许可仓库的适用范围。特别是 `GaussianModel/submodules/diff-gaussian-rasterization/` 继续受其目录内 Inria/MPII 非商业研究许可证约束。

## 3DGRUT / 3D Gaussian Ray Tracing

保留位置：`third_party/3dgrut_conversion/`

用途：仅用于本项目的 Gaussian PLY 转换链，包括 Gaussian PLY 到 `mesh.ply`、USD/USDZ 和碰撞资产的中间转换脚本。完整 3DGRUT 工程、原 README、训练/渲染主代码和无关文件未纳入本仓库。

许可证文本：`third_party_licenses/3DGRUT-Apache-2.0.txt`

```bibtex
@article{loccoz20243dgrt,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```

```bibtex
@article{wu20253dgut,
    title={3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting},
    author={Wu, Qi and Martinez Esturo, Janick and Mirzaei, Ashkan and Moenne-Loccoz, Nicolas and Gojcic, Zan},
    journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

## Gaussian Grouping

保留位置：`GaussianModel/`

用途：作为本项目独立解耦 Gaussian Splatting 模型生成、对象分割、渲染和编辑代码的重要基础。本项目在其基础上整理了面向机器人仿真场景的对象级解耦、透明度/背景处理和复现流程。

许可证文本：`third_party_licenses/GaussianGrouping-Apache-2.0.txt`

```bibtex
@inproceedings{gaussian_grouping,
    title={Gaussian Grouping: Segment and Edit Anything in 3D Scenes},
    author={Ye, Mingqiao and Danelljan, Martin and Yu, Fisher and Ke, Lei},
    booktitle={ECCV},
    year={2024}
}
```

## FastSAM

保留位置：`FastSAMRealtime/fastsam/` 与 `FastSAMRealtime/ultralytics/`

用途：用于在线 RGB 图像的快速分割，并与 Kinect 深度数据对齐生成对象点云。

许可证文本：`third_party_licenses/FastSAM-AGPL-3.0.txt`

```bibtex
@misc{zhao2023fast,
      title={Fast Segment Anything},
      author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
      year={2023},
      eprint={2306.12156},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 其他保留第三方组件

以下组件作为 `GaussianModel` 或 `Simulation` 依赖链的一部分被保留或部分保留。使用、修改和再发布时应同时遵守对应许可证文本。

| 组件 | 许可证文本 | 本项目用途 |
| --- | --- | --- |
| DEVA | `third_party_licenses/DEVA-LICENSE.md` | 2D mask/video propagation 相关流程 |
| Grounded Segment Anything | `third_party_licenses/Grounded-Segment-Anything-Apache-2.0.txt` | 文本提示分割辅助流程 |
| GroundingDINO | `third_party_licenses/GroundingDINO-LICENSE.txt` | 文本检测/定位辅助流程 |
| LaMA | `third_party_licenses/LaMA-Apache-2.0.txt` | 对象移除/补全中的 inpainting |
| Segment Anything | `third_party_licenses/Segment-Anything-LICENSE.txt` | 掩码生成与交互式分割辅助 |
| LeIsaac | `third_party_licenses/LeIsaac-Apache-2.0.txt` | IsaacLab 仿真任务、资产导入与对象状态同步 |
