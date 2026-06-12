# jixiebi-control — RM65 手臂遥操作 + 手势抓取

**版本 2.0.0** — joint1/2/3/5 四关节跟随，手腕俯仰方向已校正。

ROS 2 Humble 结课项目：PC 摄像头追踪人臂，映射到睿尔曼 RM65 机械臂（仿真 / 真机），支持竖臂激活、joint1/2/3/5 跟随、手势夹爪。

## 仓库结构

```
jixiebi-control/
├── src/rm_65_vision/     # 摄像头人臂追踪 + 手势识别
├── src/rm_65_task/       # 遥操作映射 + 机械臂控制 + launch
├── vendor/ros2_rm_robot/  # 对官方 ros2_rm_robot 的 Gazebo/配置补丁
└── scripts/apply_ros2_rm_robot_patches.sh
```

## 环境要求

- Ubuntu 22.04 + ROS 2 Humble
- Gazebo Classic 11
- MoveIt 2（RM65 配置随 ros2_rm_robot 提供）
- Python：`mediapipe`、`opencv-python`、`numpy`、`pyyaml`

## 同伴快速部署

### 1. 创建工作空间

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. 克隆本仓库与官方机械臂包

```bash
git clone git@github.com:dash-wu/jixiebi-control.git
git clone -b humble https://github.com/RealManRobot/ros2_rm_robot.git
```

### 3. 链接包到工作空间

```bash
ln -sf ~/ros2_ws/src/jixiebi-control/src/rm_65_vision .
ln -sf ~/ros2_ws/src/jixiebi-control/src/rm_65_task .
ln -sf ~/ros2_ws/src/ros2_rm_robot .
ln -sf ~/ros2_ws/src/jixiebi-control/vendor/allegro_hand_description .
ln -sf ~/ros2_ws/src/jixiebi-control/vendor/pal_urdf_utils .
```

### 4. 应用 Gazebo 场景补丁

```bash
bash ~/ros2_ws/src/jixiebi-control/scripts/apply_ros2_rm_robot_patches.sh ~/ros2_ws/src/ros2_rm_robot
```

补丁会在 RM65 末端挂载开源 **[Allegro Hand](https://github.com/pal-robotics/allegro_hand)**（PAL Robotics，Apache-2.0）。

### 5. 编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 6. 启动仿真遥操作

```bash
ros2 launch rm_65_task sim_arm_teleop_follow.launch.py
```

- **竖臂伸直**约 0.5s → 激活跟随（看摄像头窗口「跟随已激活」）
- **看 Gazebo 窗口**判断机械臂运动（RViz 仅可视化）
- 当前配置：**joint1 / joint2 / joint3 / joint5** 跟随（手腕俯仰方向与人手一致）
- 快捷键：**R** 重置，**C** 手动激活，**Q** 退出

### 7. 真机（可选）

```bash
ros2 launch rm_65_task real_arm_teleop.launch.py
```

## 调参

编辑 `src/rm_65_task/config/arm_teleop_mapping.yaml`：

- `robot_ready_pose`：机械臂竖直就绪关节角
- `active_joint_indices`：参与跟随的关节（默认 `[1,2,4]` = joint2/3/5）
- `mapping.*.scale`：跟随灵敏度

## 常见问题

| 现象 | 处理 |
|------|------|
| Gazebo 端口占用 | `killall -9 gzserver gzclient` 后重启 |
| 机械臂不动 | 确认 `ros2 control list_controllers` 两个控制器 active |
| 无法激活 | 竖臂伸直，看窗口「竖臂」指标变绿 |
| 不要用 Ctrl+Z | 会挂起所有节点，用 Ctrl+C 退出 |

## 许可证

Apache-2.0（与 RM 官方包一致）。`ros2_rm_robot` 版权归 RealManRobot。
