# RM65 手臂跟随抓取操作指南

## 动作设计（人 → 机械臂）

| 阶段 | 你的动作 | 机械臂响应 |
|------|----------|------------|
| 启动 | 抬臂伸直朝上 | 回到 `robot_ready_pose`（同款姿态） |
| 激活 | 保持朝上约 0.5s | 界面显示「跟随已激活」，开始映射 |
| 到达抓取位 | 放下/弯曲手臂 | 从就绪位出发，跟随增量移动 |
| 抓取 | 握拳 | 小幅下探 + 夹爪闭合 |
| 释放 | 张手 | 夹爪松开 |
| 复位 | HOME 手势 / **R** | 回 `robot_ready_pose`，重新等待激活 |

## 启动（仿真跟随）

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch rm_65_task sim_arm_teleop_follow.launch.py
```

## 启动（真机跟随）

```bash
ros2 launch rm_65_task real_arm_teleop.launch.py
```

## 推荐操作流程

1. 启动 Gazebo + 摄像头 + 机械臂（机械臂自动到就绪位）
2. 侧身对准摄像头，按 **1/2** 选对追踪臂
3. **抬臂伸直朝上**，看右侧「激活进度」到 100%
4. 确认「跟随已激活」后，再放下手臂去够抓取位置
5. **握拳** 抓取，**张手** 松开；**R** 重置激活

## 调参

编辑 [`config/arm_teleop_mapping.yaml`](config/arm_teleop_mapping.yaml)：

- `robot_ready_pose`：机械臂「抬臂伸直朝上」关节角（RViz/Gazebo 示教）
- `activation.*`：人体激活姿态检测阈值
- `mapping.*.scale`：跟随灵敏度
- `grasp.approach_delta`：握拳时额外下探量

## 仅调试识别（不驱动机械臂）

```bash
ros2 launch rm_65_vision arm_teleop_debug.launch.py
```
