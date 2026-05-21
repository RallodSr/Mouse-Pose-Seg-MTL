# Mouse Pose & Segmentation — Hybrid Multi-Task Learning

วิทยานิพนธ์: การนำ Multi-Task Learning มาใช้สำหรับ Instance Segmentation และ Keypoint Estimation บนภาพหนูทดลองในห้องปฏิบัติการ ด้วยโมเดลเดียว (single shared encoder)

> หมายเหตุ: การเปรียบเทียบกับ baseline ภายนอก (YOLO, DeepLabCut, SLEAP) อยู่ในส่วน paper เท่านั้น โปรเจกต์นี้เก็บเฉพาะส่วน MTL (HybridMTLNet)

---

## ภาพรวมของระบบ

| โมเดล | Architecture | Task | Metric |
|---|---|---|---|
| **HybridMTLNet** | ResNet-34 + U-Net decoder + Deconv head | Segmentation + Pose พร้อมกัน | mIoU + PCK |

**HybridMTLNet** ใช้ ResNet-34 เป็น shared encoder โดย segmentation decoder เป็น U-Net (2 channels = 2 instance masks) และ pose head เป็น deconvolutional head (6 channels = 3 keypoints × 2 ตัว) การ assign instance ใช้ **Hungarian matching** บน mask IoU ทำให้ไม่ต้องเรียนรู้ลำดับ instance ที่ตายตัว

---

## ผลการทดลอง

| Metric | Score |
|---|---|
| mIoU | **0.8105** |
| PCK@0.05 | **0.9542** |

HybridMTLNet ใช้ loss ratio seg:pose = 1:500 (seed=42) ผลเปรียบเทียบกับ baseline ภายนอกดูได้ใน `paper/main.tex`

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
├── src/                           ← MTL model (OOP)
│   ├── data/
│   │   ├── dataset.py             ← MouseMTLDataset (PyTorch Dataset)
│   │   └── prepare.py             ← prepare_dataset (Label Studio → dataset.json)
│   ├── models/
│   │   └── mtl_net.py             ← HybridMTLNet architecture
│   ├── training/
│   │   └── trainer.py             ← Trainer class (train loop, plotting)
│   ├── evaluation/
│   │   ├── metrics.py             ← calculate_miou, calculate_pck
│   │   ├── evaluator.py           ← Evaluator class (unified test eval)
│   │   └── benchmark.py           ← Benchmark class (params/FLOPs/FPS)
│   └── experiments/               ← train-and-compare studies (OOP)
│       ├── base.py                ← Experiment (base class)
│       ├── weight_ratio.py        ← WeightRatioExperiment
│       └── single_task.py         ← SingleTaskAblation
│
├── baselines/                     ← baseline models (แยกจาก MTL)
│   ├── maskrcnn.py                ← Mask R-CNN seg baseline
│   └── yolo/                      ← YOLO eval + inference scripts
│
├── app/                           ← inference scripts
│   ├── infer_images.py            ← MTL inference บนโฟลเดอร์รูป
│   └── infer_mtl_video.py         ← MTL inference บนวิดีโอ
│
├── paper/
│   ├── main.tex                   ← IEEE conference paper (LaTeX)
│   └── references.bib
│
├── config.py                      ← hyperparameters และ paths ทั้งหมด
├── main.py                        ← CLI entry point (prepare/train/eval/experiment)
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
```

---

## การ Train โมเดล

```bash
# Train ด้วย config ใน config.py
python main.py train

# Override loss weight ratio จาก CLI
python main.py train --pose-weight 500 --run-name my_run
```

hyperparameters หลักใน `config.py`:

```python
loss_seg_weight: float = 1
loss_pose_weight: float = 500   # ปรับ ratio เพื่อ balance BCE+Dice vs MSE
epochs: int = 100
lr: float = 1e-4
```

> **หมายเหตุ:** MSE heatmap loss มีขนาดเล็กกว่า BCE+Dice ประมาณ 500× โดยธรรมชาติ ต้องตั้ง `loss_pose_weight` สูงพอเพื่อให้ pose head เรียนรู้ได้

### Ablation Study

```bash
# Loss weight ratio ablation
python main.py experiment --type weight-ratio              # รัน configs แล้วเปรียบเทียบ
python main.py experiment --type weight-ratio --eval-only

# Single-task vs joint MTL ablation
python main.py experiment --type single-task               # train seg-only / pose-only / joint
python main.py experiment --type single-task --eval-only
```

single-task ablation เทรน **architecture เดียวกัน** 3 แบบ:

| Config | seg_weight | pose_weight | Metric ที่วัด |
|---|---|---|---|
| Seg-only | 1 | 0 | mIoU |
| Pose-only | 0 | 500 | PCK |
| Joint MTL | 1 | 500 | mIoU + PCK |

เพื่อแยกผลของ multi-task learning ออกจากผลของสถาปัตยกรรม (ดูผลและการอภิปรายใน `paper/main.tex`)

---

## การประเมินผล

```bash
python main.py eval --weights models/checkpoints/model_final.pth
```

**Metrics:**
- **mIoU** — per-instance IoU เฉลี่ยจาก Hungarian-matched pairs
- **PCK@0.05** — keypoint ที่ทำนายได้ภายใน 5% ของขนาดภาพ (12.8 px ที่ 256×256) ตาม Yang & Ramanan (CVPR 2013)

### Efficiency Benchmark

```bash
python main.py benchmark              # params / FLOPs / FPS (GPU + CPU)
```

วัด HybridMTLNet (1 forward pass → seg+pose) เทียบกับ modular pipeline 2 โมเดล — แสดงว่า unified ใช้ params/FLOPs ครึ่งเดียวและ FPS ~2 เท่า

---

## Inference

```bash
# บนโฟลเดอร์รูปภาพ
python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth

# บนวิดีโอ
python app/infer_mtl_video.py --input video.mp4 --weights models/checkpoints/model_final.pth --output models/inference_output/result.mp4
```

---

## สรุปขั้นตอน (Quick Reference)

```bash
# 1. ติดตั้ง
pip install -r requirements.txt

# 2. เตรียมข้อมูล
python main.py prepare

# 3. Train
python main.py train

# 4. Evaluate
python main.py eval --weights models/checkpoints/model_final.pth

# 5. Inference
python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth
```
