#!/usr/bin/env bash
# 清理残留 Gazebo，避免 "Address already in use"
killall -9 gzserver gzclient gazebo 2>/dev/null || true
sleep 2
exit 0
