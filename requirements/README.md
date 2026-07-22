# 依赖说明

本项目不建议使用单一 requirements 文件安装全部模块。Gaussian 训练、Kinect 在线处理、3DGRUT 转换和 Isaac Sim/IsaacLab 仿真依赖不同，建议分环境安装。

- `gaussian_model.txt`：`GaussianModel/` 训练、渲染、对象编辑和基础评估依赖。
- `fastsam_realtime.txt`：`FastSAMRealtime/` 在线分割、Kinect 对齐、点云裁剪与配准依赖。
- `conversion_simulation.txt`：PLY/mesh/USD 转换脚本和仿真侧辅助脚本的 Python 依赖。
- `pytorch24_reference.yml`：以 PyTorch 2.4.0 为核心的当前参考构建，不冒充历史环境导出。
- `paper_environment.md`：区分论文中实际留存的环境事实与未留存版本。

论文记录了 PyTorch 2.4 和 Kinect SDK 2.0。历史 CUDA、驱动及其余包的精确补丁版本未留存，因此不伪造锁定值。Isaac Sim/IsaacLab 与 Kinect SDK 仍需按对应运行时单独安装。

参考环境分两步安装，避免 Conda 在临时目录中错误解析相对路径：

```bash
conda env create -f requirements/pytorch24_reference.yml
conda run -n dgsrsim-pytorch24-reference pip install -r requirements/gaussian_model.txt
```
