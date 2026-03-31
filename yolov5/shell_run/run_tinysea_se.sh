#!/bin/bash
set -e

export YOLOv5_AUTOINSTALL=False
export YOLO_CONFIG_DIR=/tmp/Ultralytics

cd /workspace/Sea/yolov5

python train.py \
  --img 640 \
  --batch 2 \
  --epochs 400 \
  --data /workspace/Sea/yolov5/data/tinysea.yaml \
  --cfg /workspace/Sea/yolov5/models/yolov5s_se.yaml \
  --weights '' \
  --workers 2 \
  --project runs/train \
  --name tinysea_se