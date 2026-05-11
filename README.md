# Mouse Pose & Segmentation — Hybrid Multi-Task Learning

วิทยานิพนธ์: การนำ Multi-Task Learning มาใช้สำหรับ Instance Segmentation และ Keypoint Estimation บนภาพหนูทดลองในห้องปฏิบัติการ พร้อมเปรียบเทียบกับ YOLO baseline

---

## ภาพรวมของระบบ

ระบบประกอบด้วย 3 โมเดลที่นำมาเปรียบเทียบกัน

| โมเดล | Architecture | Task | Metric |
|---|---|---|---|
| **HybridMTLNet** (ตัวหลัก) | ResNet-34 + U-Net decoder + Deconv head | Segmentation + Keypoint พร้อมกัน | mIoU + PCK |
| YOLO-Seg baseline | YOLOv8/v9 | Segmentation เท่านั้น | mIoU |
| YOLO-Pose baseline | YOLOv8/v9 | Keypoint เท่านั้น | PCK |

**HybridMTLNet** ใช้ ResNet-34 เป็น shared encoder แชร์ feature กันระหว่างสองงาน โดย decoder ของ Segmentation เป็นแบบ U-Net (skip connections) และ decoder ของ Keypoint เป็น Deconvolutional head ที่ output เป็น Gaussian heatmap

---

## หลักการและแนวคิด (Methodology Rationale)

### ทำไมต้องเป็น Markerless

ระบบติดตามหนูทดลองแบบดั้งเดิม เช่น ANY-maze และ EthoVision ใช้ color-based tracking หรือ blob detection แบบ rule-based ซึ่งต้องติด physical marker บนตัวสัตว์หรือตั้งค่า threshold เอง วิธีเหล่านี้มีข้อจำกัดคือ ล้มเหลวง่ายเมื่อสภาพแสงเปลี่ยนหรือหนูหลายตัวซ้อนกัน และการติด marker อาจส่งผลต่อพฤติกรรมของสัตว์ทดลอง งานนี้ใช้ deep learning predict ตำแหน่ง keypoint จากภาพดิบโดยไม่ต้องแตะตัวหนู (markerless)

### ทำไมต้องเป็น Gaussian Heatmap

วิธีก่อนหน้าในยุค deep learning คือ **Direct Coordinate Regression** — ให้ CNN output ค่า (x, y) coordinates โดยตรง วิธีนี้มีปัญหาคือ loss landscape sharp ทำให้ gradient กระโดดไม่เสถียรระหว่างการเทรน และไม่สามารถ encode ความไม่แน่นอนของตำแหน่งได้

Newell et al. (Stacked Hourglass Networks, ECCV 2016) เสนอการแทน coordinate ด้วย **2D Gaussian probability map** (heatmap) แทน ซึ่งให้ gradient เรียบกว่า เทรนเสถียรกว่า และสามารถแสดงความไม่แน่นอนของตำแหน่งได้ผ่านการกระจายตัวของ Gaussian Mathis et al. (DeepLabCut, Nature Neuroscience 2018) นำแนวคิดนี้มาประยุกต์ใช้กับ lab animal โดยเฉพาะ

ในงานนี้ heatmap ถูกใช้ใน 3 ขั้นตอน:
1. **Dataset** (`src/data/dataset.py`) — แปลง annotation coordinates → Gaussian heatmap เป็น training target
2. **Training** (`src/training/trainer.py`) — MSELoss ระหว่าง predicted heatmap กับ GT heatmap
3. **Evaluation** (`src/evaluation/metrics.py`) — หา argmax ของ predicted heatmap เป็นตำแหน่ง keypoint แล้ววัด pixel distance กับ GT

### ทำไมเลือก 3 Keypoints (Nose, Shoulder, Tail)

สามจุดนี้เพียงพอสำหรับการวิเคราะห์พฤติกรรมหลักของหนูทดลอง:
- **Nose** — ทิศทางที่หนูหันหน้า (heading direction)
- **Shoulder** — ตำแหน่งกลางลำตัว (body position)
- **Tail** — orientation ของลำตัว (body axis)

ครบพอให้วิเคราะห์ locomotion, turning, freezing behavior โดยไม่ต้อง annotate จุดที่มองเห็นยากอย่างหูหรือขา ซึ่งเพิ่ม annotation error มากกว่าจะเพิ่ม information ที่เป็นประโยชน์

### ทำไมใช้ PCK แทน mAP (OKS)

งานนี้ใช้ **PCK (Percentage of Correct Keypoints)** ตาม Yang & Ramanan (CVPR 2013) โดยกำหนด threshold ที่ 5% ของขนาดภาพ (12.8 px ที่ resolution 256×256) ซึ่งสอดคล้องกับแนวทางในงาน lab animal pose estimation (Mathis et al., 2018)

แม้ DeepLabCut 3.0 จะเปลี่ยนไปใช้ OKS-based mAP (COCO standard) แต่ OKS ต้องการ per-instance predictions และ per-keypoint sigma ที่ estimate จาก dataset ขนาดใหญ่ ซึ่งไม่เหมาะกับขนาด dataset และ architecture ของงานนี้ที่ใช้ combined heatmap representation

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
├── notebooks/
│   ├── 01_data_exploration.ipynb  ← EDA: distribution, visibility, sample viz
│   └── 02_results_visualization.ipynb ← training curves + qualitative comparison
│
├── src/                           ← core library (import ได้)
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
├── app/                           ← inference scripts (รัน demo)
│   ├── infer_images.py            ← MTL inference บนโฟลเดอร์รูป
│   ├── infer_mtl_video.py         ← MTL inference บนวิดีโอ
│   ├── infer_yolo_pose_video.py   ← YOLO-Pose inference บนวิดีโอ
│   └── infer_yolo_seg_video.py    ← YOLO-Seg inference บนวิดีโอ
│
├── config.py                      ← hyperparameters และ paths ทั้งหมด
├── main.py                        ← CLI entry point
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU (แนะนำ VRAM ≥ 6GB สำหรับ batch size 16)
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

### รูปแบบข้อมูลที่ต้องการ

ระบบรองรับข้อมูลที่ annotate ด้วย **Label Studio** โดยต้องมีไฟล์ดังนี้

```
data/
├── images/           ← ภาพต้นฉบับ (ชื่อไฟล์ต้องตรงกับใน JSON)
├── masks/            ← binary mask รูปแบบ PNG (สีขาว = ตัวหนู, สีดำ = พื้นหลัง)
│   ├── dl0000_1.png  ← naming: <image_name>_<mouse_index>.png
│   └── dl0000_2.png
└── label_studio_export.json  ← export จาก Label Studio (JSON format)
```

**Keypoints ที่ annotate:** `nose`, `shoulder`, `tail` (3 จุดต่อตัว)

### แปลงข้อมูลเป็น dataset กลาง

```bash
python main.py prepare
```

คำสั่งนี้จะ:
1. อ่าน `data/label_studio_export.json` และจับคู่ keypoint กับ mask ของหนูแต่ละตัว
2. แบ่งข้อมูล train / val / test (80% / 10% / 10%)
3. บันทึกผลที่ `data/dataset.json`

ตัวอย่างผลลัพธ์:
```
Split → Train: 160 | Val: 20 | Test: 20
```

### แปลงข้อมูลสำหรับ YOLO baseline

```bash
# สำหรับ YOLO-Pose
python main.py convert --task pose
# → data/yolo_pose/

# สำหรับ YOLO-Seg
python main.py convert --task seg
# → data/yolo_seg/
```

---

## การ Train โมเดล

### MTL Model (HybridMTLNet)

**Option A — รัน experiment เดียว**

ปรับ hyperparameters ได้ที่ `config.py` ก่อน train

```python
# config.py — ค่าสำคัญที่ควรปรับ
@dataclass
class TrainConfig:
    batch_size: int = 16       # ลดลงถ้า VRAM ไม่พอ
    epochs: int = 100
    lr: float = 1e-4
    loss_seg_weight: float = 0.5
    loss_pose_weight: float = 5000.0   # ปรับเพื่อ balance CE vs MSE
    pck_threshold: float = 0.05        # 5% ของ image size = 12.8 px
```

> **หมายเหตุ `loss_pose_weight`:** MSE loss มีค่าเล็กกว่า CE loss มาก การคูณ 5000 ทำให้ทั้งสอง task มี gradient ในระดับเดียวกัน ถ้า PCK ต่ำให้เพิ่มค่านี้ ถ้า mIoU ต่ำให้ลดลง

```bash
python main.py train
```

หรือ override weight ratio ตรงจาก CLI โดยไม่ต้องแก้ config:

```bash
python main.py train --seg-weight 1 --pose-weight 40 --run-name baseline_1_40
python main.py train --seg-weight 1 --pose-weight 80 --run-name baseline_1_80
```

**Option B — รันทุก experiment แล้วเปรียบเทียบอัตโนมัติ (แนะนำ)**

```bash
python run_experiments.py
```

รันครบทั้ง 4 config ต่อเนื่องกัน:

| Config | seg_weight | pose_weight | Output dir |
|---|---|---|---|
| Ours | 1 | 1 | `models/checkpoints/ours_w1_1/` |
| Baseline A | 1 | 40 | `models/checkpoints/baseline_1_40/` |
| Baseline B | 1 | 80 | `models/checkpoints/baseline_1_80/` |
| Baseline C | 1 | 100 | `models/checkpoints/baseline_1_100/` |

หลัง train เสร็จ ระบบจะ evaluate test set ทุก config และสร้าง:
- `models/experiment_results.json` — ตาราง mIoU + PCK ทุก experiment
- `models/experiment_comparison.png` — bar chart เปรียบเทียบ

ถ้าเทรนไว้แล้วและต้องการแค่ดูผล:
```bash
python run_experiments.py --eval-only
```

ผลลัพธ์จะถูกบันทึกที่ `models/checkpoints/`
```
models/checkpoints/
├── checkpoint_epoch_010.pth
├── checkpoint_epoch_020.pth
├── ...
├── model_final.pth
└── training_curves.png
```

### YOLO Baseline

```bash
# YOLO-Pose
yolo pose train data=data/yolo_pose/dataset.yaml model=yolo26m-pose.pt epochs=100 imgsz=256 amp=False

# YOLO-Seg
yolo segment train data=data/yolo_seg/dataset_seg.yaml model=yolo26m-seg.pt epochs=100 imgsz=256 amp=False
```

Weights จะอยู่ที่ `runs/pose/train/weights/best.pt` และ `runs/segment/train/weights/best.pt`

---

## การประเมินผล

### MTL Model

```bash
python main.py eval --weights models/checkpoints/model_final.pth
```

```
Test mIoU : 0.8412
Test PCK  : 0.7834
```

### YOLO Baseline

```bash
# YOLO-Seg → mIoU
python main.py eval-yolo \
  --task seg \
  --weights runs/segment/train/weights/best.pt

# YOLO-Pose → PCK
python main.py eval-yolo \
  --task pose \
  --weights runs/pose/train/weights/best.pt
```

### คำนิยาม Metric

**mIoU (Mean Intersection over Union)** — วัดความแม่นยำของ Segmentation

$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c}$$

โดย $C$ = จำนวน class (background + mouse = 2)

**PCK (Percentage of Correct Keypoints)** — วัดความแม่นยำของ Keypoint

$$\text{PCK} = \frac{\text{จำนวนจุดที่ทำนายได้ภายใน threshold}}{\text{จำนวนจุดทั้งหมดที่มองเห็น}}$$

threshold = 5% ของขนาดภาพ = 12.8 px (ที่ resolution 256×256)

---

## Inference

### บนโฟลเดอร์รูปภาพ

```bash
python app/infer_images.py \
  --input data/images \
  --weights models/checkpoints/model_final.pth \
  --output models/inference_output
```

ผลลัพธ์: ภาพที่มี **สีเขียว overlay** ตรง segmentation mask และ **จุดสี** ตรง keypoint

| สี | ความหมาย |
|---|---|
| เขียว (overlay) | Segmentation mask |
| แดง | Keypoint: Nose |
| ส้ม | Keypoint: Shoulder |
| น้ำเงิน | Keypoint: Tail |

### บนวิดีโอ

```bash
# MTL Model
python app/infer_mtl_video.py \
  --input lab_video.mp4 \
  --weights models/checkpoints/model_final.pth \
  --output models/inference_output/result.mp4

# YOLO-Pose
python app/infer_yolo_pose_video.py \
  --input lab_video.mp4 \
  --weights runs/pose/train/weights/best.pt

# YOLO-Seg
python app/infer_yolo_seg_video.py \
  --input lab_video.mp4 \
  --weights runs/segment/train/weights/best.pt
```

---

## Notebooks

ต้องติดตั้ง Jupyter ก่อน:

```bash
pip install jupyter ipykernel
jupyter notebook notebooks/
```

| Notebook | เนื้อหา |
|---|---|
| `01_data_exploration.ipynb` | สำรวจ dataset: จำนวนหนูต่อภาพ, keypoint visibility, sample visualization |
| `02_results_visualization.ipynb` | training curves, qualitative comparison ระหว่าง GT กับ prediction |

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
yolo pose train data=data/yolo_pose/dataset.yaml model=yolo26m-pose.pt epochs=100 imgsz=256 amp=False
yolo segment train data=data/yolo_seg/dataset_seg.yaml model=yolo26m-seg.pt epochs=100 imgsz=256 amp=False

# 4. Evaluate
python main.py eval --weights models/checkpoints/model_final.pth
python main.py eval-yolo --task seg --weights runs/segment/train/weights/best.pt
python main.py eval-yolo --task pose --weights runs/pose/train/weights/best.pt

# 5. Inference
python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth
```

---

## การ Customize

### เพิ่มจำนวน Keypoints

แก้ที่ `config.py`:

```python
@dataclass
class DataConfig:
    keypoint_names: list = field(default_factory=lambda: ["nose", "ear_l", "ear_r", "shoulder", "hip", "tail_base", "tail_tip"])

@dataclass
class ModelConfig:
    num_keypoints: int = 7   # ต้องตรงกับ len(keypoint_names)
```

### เปลี่ยน Backbone

แก้ที่ `src/models/mtl_net.py` บรรทัด `models.resnet34(...)` เป็น `resnet50` หรือ `resnet18` ได้เลย พร้อมปรับ channel sizes ใน decoder ให้ตรงกัน

---

## Dependencies หลัก

| Package | Version | ใช้สำหรับ |
|---|---|---|
| torch | ≥ 2.0.0 | Deep learning framework |
| torchvision | ≥ 0.15.0 | ResNet backbone, transforms |
| ultralytics | ≥ 8.0.0 | YOLO baseline |
| opencv-python | ≥ 4.8.0 | Image processing, heatmap extraction |
| scikit-learn | ≥ 1.3.0 | train_test_split |
| matplotlib | ≥ 3.7.0 | Training curves |
