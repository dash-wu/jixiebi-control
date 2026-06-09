# 遥操作模式说明

## 当前模式：`joints`（默认）

人体关节增量 → RM65 关节映射，由以下模块实现：

- `rm_65_vision/arm_teleop_tracker.py`：MediaPipe world landmarks + swing-twist 解耦
- `rm_65_task/teleop_mapper.py`：增量映射 + 可选标定 scale/offset

## 可选升级（未默认启用）

### pose_landmarker_heavy

- 将 Holistic 拆分为 `PoseLandmarker`（heavy）+ `HandLandmarker`
- 预期：pose 关键点略稳，J1 改善约 5~10%
- 成本：需重写 launch 与 tracker 初始化，失去 face/holistic 一体流程

### 手腕 IK 模式 `teleop_mode: ik`

- 用 world 坐标估计手腕位置 + 前臂方向
- RM65 数值 IK 解 joint1~3，绕过人体角→机器角映射
- 成本：3~5 天，需 URDF/FK/IK 与奇异点处理
- 配置项 `teleop_mode: ik` 已在 `arm_teleop_mapping.yaml` 预留，当前自动回退到 `joints`

## 使用建议

1. 优先使用 **world landmarks + 三点标定（K 键）**
2. 正面对摄像头时 J1 精度较弱，建议身体转 30~45°
3. 每次改参数后 **R 重置 + 竖臂激活**
