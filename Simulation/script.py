'''

dependencies\IsaacLab\isaaclab.bat -p tools\create_hinged_usd_assembly.py `
  --part-a-usdz assets\hinge_case\microwave\microwave_A_invis.usdz `
  --part-b-usdz assets\hinge_case\microwave\microwave_B_invis.usdz `
  --split-json assets\hinge_case\microwave\picked_hinge_split.json `
  --output assets\hinge_case\microwave\microwave_invis_hinged_assembly.usdz `
  --fixed-body A `
  --lower-limit-deg 0 `
  --upper-limit-deg 90

dependencies\IsaacLab\isaaclab.bat -p tools\create_hinged_usd_assembly.py `
  --part-a-usdz assets\hinge_case\medicine_cabinet\medicine_cabinet_A_vis.usdz `
  --part-b-usdz assets\hinge_case\medicine_cabinet\medicine_cabinet_B_vis.usdz `
  --split-json assets\hinge_case\medicine_cabinet\picked_hinge_split.json `
  --output assets\hinge_case\medicine_cabinet\medicine_cabinet_vis_hinged_assembly.usdz `
  --fixed-body A `
  --lower-limit-deg 0 `
  --upper-limit-deg 90


dependencies\IsaacLab\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent_hinged.py `
  --task=LeIsaac-SO101-PickUnkownObject-v0 `
  --teleop_device=so101leader  `
  --port=COM3 `
  --num_envs=1 `
  --device=cpu `
  --enable_cameras `
  --record `
  --dataset_file=.\datasets\dataset03.hdf5  --debug-leader --debug-leader-period 0.2















'''