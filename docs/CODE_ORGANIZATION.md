# 发布版代码整理说明

整理日期：2026-06-02；论文口径同步更新：2026-07-09

对应论文题名：`DGSRSim: An Object-Level Decoupled 3D Gaussian Scene Representation for Robot Simulation and Online State Synchronization`

输入目录：`code/`

输出目录：`DGSRSim_GitHub_Release_20260602_upload/`

## 整理原则

- 不修改原始 `code/` 目录。
- 不复制任何 `.git`、`.github`、`.gitmodules`、IDE 配置、缓存或内部协作文件。
- 不复制权重、checkpoint、点云、图片、视频、压缩包和仿真大资产。
- 原始第三方 README 不作为发布版主文档保留，统一合并进根目录 `README.md` 和 `THIRD_PARTY_NOTICES.md`。
- 3DGRUT 只保留 PLY 到 mesh/USD/USDZ 转换链相关脚本，不保留完整项目。
- GaussianModel 保留主体训练、渲染、分割、编辑、LaMA/DEVA 辅助代码和 CUDA 子模块源码，以保证复现路径可追踪。
- FastSAMRealtime 保留 Kinect 采集、RGB-D 对齐、FastSAM 推理、对象点云裁剪、滤波和配准所需源码。
- Simulation 保留 LeIsaac/IsaacLab 任务、脚本和工具源码，资产目录只保留 `.gitkeep`。该模块对应论文中的仿真端对象状态写入、可组合场景和交互接口记录，不作为机器人任务可靠性或抓取成功率证据。

## 论文一致性说明

- `GaussianModel/` 对应离线对象 Gaussian 资产和背景 Gaussian 场构建。
- `FastSAMRealtime/` 对应在线 RGB-D 观测、对象点云构造、配准质量过滤和粗到细位姿估计。
- `third_party/3dgrut_conversion/` 对应 Gaussian PLY 到 mesh/USD/USDZ/碰撞资产的转换链。
- `Simulation/` 对应对象资产导入、场景组合、机器人资产接入和对象状态同步。
- 当前发布仓库不包含大模型权重、原始传感器采集、训练输出、点云和 USD/USDZ 大资产。

## 当前发布副本规模

| 模块 | 文件数 | 大小 MB |
| --- | ---: | ---: |
| `GaussianModel/` | 1167 | 5.46 |
| `FastSAMRealtime/` | 182 | 1.51 |
| `Simulation/` | 176 | 0.99 |
| `third_party/` | 22 | 0.15 |
| `third_party_licenses/` | 9 | 0.11 |

最终上传前应再次运行检查，确认没有大文件或内部文件进入 Git。
