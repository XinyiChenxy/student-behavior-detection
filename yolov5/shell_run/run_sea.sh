#!/bin/bash
set -e

export YOLOv5_AUTOINSTALL=False
export YOLO_CONFIG_DIR=/tmp/Ultralytics

cd /workspace/Sea/yolov5

python train.py \
  --img 640 \
  --batch 8 \
  --epochs 400 \
  --data /workspace/Sea/yolov5/data/seadrones.yaml \
  --weights yolov5s.pt \
  --workers 4 \
  --project runs/train \
  --name sea_baseline

