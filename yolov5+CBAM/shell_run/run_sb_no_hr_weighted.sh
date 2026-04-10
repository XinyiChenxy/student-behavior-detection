#!/bin/bash
set -e

export YOLOv5_AUTOINSTALL=False
export YOLO_CONFIG_DIR=/tmp/Ultralytics

cd /workspace/yolov5

python train_weighted_loss.py \
  --img 640 \
  --batch 2 \
  --epochs 1000 \
  --data data/StudentWatch_no_hr.yaml \
  --cfg models/yolov5s_cbam.yaml \
  --weights yolov5s.pt \
  --workers 2 \
  --cache \
  --project runs/train \
  --name sb_cbam_no_hr_weighted \
  --patience 0