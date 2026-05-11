# Mouse Pose & Segmentation — Hybrid Multi-Task Learning

วิทยานิพนธ์: การนำ Multi-Task Learning มาใช้สำหรับ Instance Segmentation และ Keypoint Estimation บนภาพหนูทดลองในห้องปฏิบัติการ พร้อมเปรียบเทียบกับ YOLO baseline

---

## ภาพรวมของระบบ

| โมเดล | Architecture | Task | Metric |
|---|---|---|---|
| **HybridMTLNet** (ตัวหลัก) | ResNet-34 + U-Net decoder + Deconv head | Segmentation + Pose พร้อมกัน | mIoU + PCK |
| YOLO-Seg baseline | YOLOv8m-seg | Instance Segmentation เท่านั้น | mIoU |
| YOLO-Pose baseline | YOLOv8m-pose | Keypoint เท่านั้น | PCK |

**HybridMTLNet** ใช้ ResNet-34 เป็น shared encoder โดย segmentation decoder เป็น U-Net (2 channels = 2 instance masks) และ pose head เป็น deconvolutional head (6 channels = 3 keypoints × 2 ตัว) การ assign instance ใช้ **Hungarian matching** บน mask IoU ทำให้ไม่ต้องเรียนรู้ลำดับ instance ที่ตายตัว

---

## ผลการทดลอง

| Model | mIoU | PCK@0.05 |
|---|---|---|
| YOLOv8m-Seg | 0.5383 | — |
| YOLOv8m-Pose | — | 0.9105 |
| **HybridMTLNet (Ours)** | **0.8113** | **0.9621** |
| Δ vs. YOLO baseline | +0.2730 | +0.0516 |

---

## โครงสร้างโปรเจกต์

```
Mouse-Pose-Seg-MTL/
│
├── data/                          ← raw images, masks, annotations  [gitignored]
│   ├── images/                    ← ภาพต้นฉบับ (.png / .jpg)
│   ├── masks/                     ← binary mask ของหนูแต่ละตัว
│   ├── label_studio_export.json   ← export จาก Label Studio
│   └── dataset.json               ← dataset กลางที่สร้างจาก prepare
│
├── models/                        ← saved weights & checkpoints      [gitignored]
│   ├── checkpoints/               ← ผลจากการ train MTL
│   └── inference_output/          ← ผลจาก inference
│
├── src/
│   ├── data/
│   │   ├── dataset.py             ← MouseMTLDataset (PyTorch Dataset)
│   │   └── prepare.py             ← prepare_dataset, convert_yolo_pose/seg
│   ├── models/
│   │   └── mtl_net.py             ← HybridMTLNet architecture
│   ├── training/
│   │   └── trainer.py             ← Trainer class (train loop, plotting)
│   └── evaluation/
│       ├── metrics.py             ← calculate_miou, calculate_pck
│       └── yolo.py                ← evaluate_yolo_miou, evaluate_yolo_pck
│
├── app/                           ← inference scripts
│   ├── infer_images.py            ← MTL inference บนโฟลเดอร์รูป
│   ├── infer_mtl_video.py         ← MTL inference บนวิดีโอ
│   ├── infer_yolo_pose_video.py   ← YOLO-Pose inference บนวิดีโอ
│   └── infer_yolo_seg_video.py    ← YOLO-Seg inference บนวิดีโอ
│
├── paper/
│   ├── main.tex                   ← IEEE conference paper (LaTeX)
│   └── references.bib
│
├── config.py                      ← hyperparameters และ paths ทั้งหมด
├── main.py                        ← CLI entry point
├── run_experiments.py             ← รัน ablation study (loss weight ratio)
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU (แนะนำ VRAM ≥ 6 GB สำหรับ batch size 16)
- PyTorch 2.0+

---

## การติดตั้ง

```bash
# 1. Clone repository
git clone <repo-url>
cd Mouse-Pose-Seg-MTL

# 2. สร้าง virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. ติดตั้ง dependencies
pip install -r requirements.txt
```

---

## การเตรียมข้อมูล

โครงสร้างข้อมูลที่ต้องการ:

```
data/
├── images/           ← ภาพต้นฉบับ
├── masks/            ← binary mask (naming: <image_name>_<mouse_index>.png)
└── label_studio_export.json
```

**Keypoints:** `nose`, `shoulder`, `tail` (3 จุดต่อตัว, สูงสุด 2 ตัวต่อภาพ)

```bash
# แปลง Label Studio export → dataset.json (แบ่ง 80/10/10)
python main.py prepare

# แปลงสำหรับ YOLO baseline
python main.py convert --task pose   # → data/yolo_pose/
python main.py convert --task seg    # → data/yolo_seg/
```

---

## การ Train โมเดล

### MTL Model (HybridMTLNet)

```bash
# Train ด้วย config ใน config.py
python main.py train

# Override loss weight ratio จาก CLI
python main.py train --pose-weight 400 --run-name my_run
```

hyperparameters หลักใน `config.py`:

```python
loss_seg_weight: float = 1
loss_pose_weight: float = 400   # ปรับ ratio เพื่อ balance BCE+Dice vs MSE
epochs: int = 100
lr: float = 1e-4
```

> **หมายเหตุ:** MSE heatmap loss มีขนาดเล็กกว่า BCE+Dice ประมาณ 500× โดยธรรมชาติ ต้องตั้ง `loss_pose_weight` สูงพอเพื่อให้ pose head เรียนรู้ได้

### Ablation Study (Loss Weight Ratio)

```bash
python run_experiments.py           # รัน 4 configs แล้วเปรียบเทียบอัตโนมัติ
python run_experiments.py --eval-only  # ประเมินจาก checkpoint ที่มีอยู่
```

| Config | pose_weight | Output dir |
|---|---|---|
| 1:1 | 1 | `models/checkpoints/1_1/` |
| 1:100 | 100 | `models/checkpoints/1_100/` |
| **1:400 (ใช้จริง)** | **400** | **`models/checkpoints/1_400/`** |
| 1:800 | 800 | `models/checkpoints/1_800/` |

### YOLO Baseline

```bash
yolo pose train data=data/yolo_pose/dataset.yaml model=yolov8m-pose.pt epochs=100 imgsz=256 amp=False
yolo segment train data=data/yolo_seg/dataset_seg.yaml model=yolov8m-seg.pt epochs=100 imgsz=256 amp=False
```

---

## การประเมินผล

```bash
# MTL model
python main.py eval --weights models/checkpoints/model_final.pth

# YOLO baselines
python main.py eval-yolo --task seg  --weights runs/segment/train/weights/best.pt
python main.py eval-yolo --task pose --weights runs/pose/train/weights/best.pt
```

**Metrics:**
- **mIoU** — per-instance IoU เฉลี่ยจาก Hungarian-matched pairs
- **PCK@0.05** — keypoint ที่ทำนายได้ภายใน 5% ของขนาดภาพ (12.8 px ที่ 256×256) ตาม Yang & Ramanan (CVPR 2013)

---

## Inference

```bash
# บนโฟลเดอร์รูปภาพ
python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth

# บนวิดีโอ (MTL)
python app/infer_mtl_video.py --input video.mp4 --weights models/checkpoints/model_final.pth --output models/inference_output/result.mp4

# บนวิดีโอ (YOLO)
python app/infer_yolo_pose_video.py --input video.mp4 --weights runs/pose/train/weights/best.pt
python app/infer_yolo_seg_video.py  --input video.mp4 --weights runs/segment/train/weights/best.pt
```

---

## สรุปขั้นตอน (Quick Reference)

```bash
# 1. ติดตั้ง
pip install -r requirements.txt

# 2. เตรียมข้อมูล
python main.py prepare
python main.py convert --task pose
python main.py convert --task seg

# 3. Train
python main.py train
yolo pose train data=data/yolo_pose/dataset.yaml model=yolov8m-pose.pt epochs=100 imgsz=256 amp=False
yolo segment train data=data/yolo_seg/dataset_seg.yaml model=yolov8m-seg.pt epochs=100 imgsz=256 amp=False

# 4. Evaluate
python main.py eval --weights models/checkpoints/model_final.pth
python main.py eval-yolo --task seg  --weights runs/segment/train/weights/best.pt
python main.py eval-yolo --task pose --weights runs/pose/train/weights/best.pt

# 5. Inference
python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth
```
