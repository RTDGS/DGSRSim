# 大文件外置清单

本仓库不直接上传模型权重、训练输出、点云、视频、图片和仿真资产。大型资产、原始采集数据、checkpoint 和第三方权重按论文的数据可用性声明，通过补充材料包、审稿包、源项目下载地址或许可允许的归档链接提供。

外部资产包：当前 GitHub 仓库不内置大文件；公开归档或审稿包链接确定后在此处补充。

## 下载后放置规则

| 类别 | 下载后放置路径 | 说明 |
| --- | --- | --- |
| Gaussian 训练数据和输出 | `GaussianModel/data/`、`GaussianModel/output/`、`GaussianModel/result/`、`GaussianModel/checkpoint/` | 训练输入、checkpoint、渲染结果和对象点云 |
| DEVA/SAM/GroundingDINO 权重 | `GaussianModel/Tracking-Anything-with-DEVA/saves/` | 伪标签和对象编辑流程所需权重 |
| LaMA 权重 | `GaussianModel/lama/big-lama/`、`GaussianModel/lama/hub/checkpoints/` | inpainting 流程所需权重 |
| FastSAM 权重 | `FastSAMRealtime/weights/` | 在线分割流程所需权重，例如 `FastSAM-x.pt` |
| 在线点云输出 | `FastSAMRealtime/rt_ply_out/` | Kinect/FastSAM 运行产生的示例或中间 PLY |
| 仿真资产 | `Simulation/assets/` | USD/USDZ、机器人、场景、碰撞和 Gaussian splat 仿真资产 |

## 关键大文件示例

以下路径来自原始 `code/` 目录，用于说明哪些内容被外置。GitHub 发布副本中不包含这些文件。

| 原始路径 | 大小 MB | 下载后目标位置 |
| --- | ---: | --- |
| `code/GaussianModel/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth` | 2445.75 | `GaussianModel/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth` |
| `code/GaussianModel/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth` | 661.85 | `GaussianModel/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth` |
| `code/GaussianModel/Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth` | 264.08 | `GaussianModel/Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth` |
| `code/GaussianModel/lama/big-lama/models/best.ckpt` | 391.05 | `GaussianModel/lama/big-lama/models/best.ckpt` |
| `code/GaussianModel/lama/hub/checkpoints/vgg16-397923af.pth` | 527.80 | `GaussianModel/lama/hub/checkpoints/vgg16-397923af.pth` |
| `code/GaussianModel/checkpoint/lerf_mask/figurines/point_cloud/iteration_30000/point_cloud.ply` | 1070.68 | `GaussianModel/checkpoint/lerf_mask/figurines/point_cloud/iteration_30000/point_cloud.ply` |
| `code/GaussianModel/checkpoint/lerf_mask/teatime/point_cloud/iteration_30000/point_cloud.ply` | 817.20 | `GaussianModel/checkpoint/lerf_mask/teatime/point_cloud/iteration_30000/point_cloud.ply` |
| `code/GaussianModel/checkpoint/lerf_mask/ramen/point_cloud/iteration_30000/point_cloud.ply` | 376.73 | `GaussianModel/checkpoint/lerf_mask/ramen/point_cloud/iteration_30000/point_cloud.ply` |
| `code/FastSAM/weights/FastSAM-x.pt` | 138.23 | `FastSAMRealtime/weights/FastSAM-x.pt` |
| `code/FastSAM/rt_ply_out/astronaut.ply` | 25.98 | `FastSAMRealtime/rt_ply_out/astronaut.ply` |
| `code/leisaac/assets/scenes/kitchen_with_orange.zip` | 69.54 | `Simulation/assets/scenes/kitchen_with_orange.zip` |
| `code/leisaac/assets/scenes/kitchen_with_orange/scene.usd` | 37.92 | `Simulation/assets/scenes/kitchen_with_orange/scene.usd` |
| `code/leisaac/assets/hinge_case/table/table_vis.usdz` | 175.25 | `Simulation/assets/hinge_case/table/table_vis.usdz` |
| `code/leisaac/assets/hinge_case/table/table_invis.usdz` | 175.25 | `Simulation/assets/hinge_case/table/table_invis.usdz` |
| `code/leisaac/assets/robots/so101_follower.usd` | 22.16 | `Simulation/assets/robots/so101_follower.usd` |

## 统计

原始 `code/` 目录中，排除 `.git` 后超过 10 MB 的文件按类别统计如下：

| 类别 | 文件数 | 合计 MB |
| --- | ---: | ---: |
| 模型权重 | 9 | 4835.54 |
| Gaussian 输出或数据 | 6 | 4205.80 |
| FastSAM 权重 | 1 | 138.23 |
| 在线点云输出 | 4 | 92.17 |
| 仿真资产 | 13 | 598.10 |
| 其他媒体 | 1 | 12.67 |
