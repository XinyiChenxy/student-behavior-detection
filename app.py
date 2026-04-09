from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import sys
import cv2
import numpy as np
import base64

# 把 yolov5 加入路径
sys.path.insert(0, './yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

app = FastAPI()

# ── 载入模型 ────────────────────────────────────────────────
model = DetectMultiBackend(
    './yolov5/runs/train/sb_no_hr_aug/weights/best.pt',
    device=torch.device('cpu')
)
model.eval()

CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.45

# 类别颜色
COLORS = {
    "bowing_the_head":          (0, 165, 255),
    "learning_over_the_table":  (0, 0, 255),
    "reading":                  (0, 255, 0),
    "using_phone":              (255, 0, 0),
    "writing":                  (255, 255, 0),
}

# ── 推理函数 ────────────────────────────────────────────────
def run_inference(image_bytes):
    # 解码图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img0  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 预处理
    img = letterbox(img0, 640, stride=model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred = model(img)
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)

    # 画框 + 收集结果
    results = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                class_name = model.names[int(cls)]
                label      = f"{class_name.replace('_', ' ')} {conf:.2f}"
                color      = COLORS.get(class_name, (0, 255, 0))

                results.append({
                    "label":      class_name.replace('_', ' '),
                    "english":    class_name,
                    "confidence": f"{conf * 100:.1f}%"
                })

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img0, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 编码回 base64
    _, buffer  = cv2.imencode('.jpg', img0)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return results, img_base64

# ── 路由 ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    detections, img_b64 = run_inference(contents)
    return JSONResponse({
        "detections": detections,
        "image":      img_b64
    })