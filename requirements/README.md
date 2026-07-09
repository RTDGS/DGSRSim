# 依赖说明

本项目不建议使用单一 requirements 文件安装全部模块。Gaussian 训练、Kinect 在线处理、3DGRUT 转换和 Isaac Sim/IsaacLab 仿真依赖不同，建议分环境安装。

- `gaussian_model.txt`：`GaussianModel/` 训练、渲染、对象编辑和基础评估依赖。
- `fastsam_realtime.txt`：`FastSAMRealtime/` 在线分割、Kinect 对齐、点云裁剪与配准依赖。
- `conversion_simulation.txt`：PLY/mesh/USD 转换脚本和仿真侧辅助脚本的 Python 依赖。

CUDA、PyTorch、Kinect SDK、Isaac Sim 和 IsaacLab 版本需要按本机系统单独匹配。

