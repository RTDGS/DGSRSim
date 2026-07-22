# Paper and Reference Environments

## Recorded experimental facts

- GPU: two NVIDIA RTX 4090 devices.
- Deep-learning framework: PyTorch 2.4.
- Offline random seed: 0 for Python, NumPy, and PyTorch.
- RGB-D interface: Kinect v2 with Kinect SDK 2.0.
- Simulation stack retained by the released LeIsaac package metadata: Isaac Sim
  5.1.0 and IsaacLab 2.3.0.

The archived experiment record does not contain exact Python, CUDA toolkit,
NVIDIA driver, Open3D, NumPy, or auxiliary-package patch versions. This release
does not invent those values. `pytorch24_reference.yml` provides a current,
machine-readable PyTorch 2.4 reference base for rebuilding the Gaussian module;
it is not represented as a byte-identical historical environment export.

The simulation environment remains separate because Isaac Sim manages its own
Python runtime. The Kinect SDK is installed at operating-system level and is not
distributed through pip or conda.
