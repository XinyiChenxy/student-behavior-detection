#!/bin/bash
set -e

export YOLOv5_AUTOINSTALL=False
export YOLO_CONFIG_DIR=/tmp/Ultralytics

cd /workspace/yolov5

python train.py \
  --img 640 \
  --batch 2 \
  --epochs 100 \
  --data data/StudentWatch.yaml \
  --cfg models/yolov5s_cbam.yaml \
  --weights yolov5s.pt \
  --workers 2 \
  --cache \
  --project runs/train \
  --name sb_cbam