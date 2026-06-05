#!/usr/bin/env bash
# 将本仓库 vendor 中的修改覆盖到 ros2_rm_robot（需先 clone humble 分支）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${ROOT}/vendor/ros2_rm_robot"
TARGET="${1:-${HOME}/ros2_rm_robot}"

if [[ ! -d "${TARGET}/.git" ]]; then
  echo "错误: 未找到 ros2_rm_robot 仓库: ${TARGET}"
  echo "请先执行:"
  echo "  git clone -b humble https://github.com/RealManRobot/ros2_rm_robot.git ${TARGET}"
  exit 1
fi

echo "应用补丁 -> ${TARGET}"
rsync -av "${VENDOR}/" "${TARGET}/"
echo "完成。请重新编译: cd ~/ros2_ws && colcon build"
