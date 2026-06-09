# ============================================================
#  BTF YOLOv8m PHASE 1  —  VS CODE / LOCAL MACHINE VERSION
#  Compatible: torch>=2.0 | ultralytics latest
# ============================================================
#
#  FOLDER STRUCTURE (script sits at the root):
#
#  Root/
#  ├── Raw Images/
#  │   └── Location 1/
#  │       ├── Double Lane/
#  │       │   └── 1-LOC1-0700-0800-DL/   ← .jpg / .jpeg / .png files
#  │       └── Single Lane/ ...
#  ├── Annotated Images/
#  │   └── Location 1/
#  │       ├── Double Lane/
#  │       │   └── 1-LOC1-0700-0800-DL/   ← .xml files (same sub-path)
#  │       └── Single Lane/ ...
#  ├── yolov8m.py          ← this script
#  └── requirements.txt
#
#  Just run:  python yolov8m.py
# ============================================================

import os
import random
import hashlib
import glob
import shutil
import collections
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import ultralytics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
#  CONFIG
# ============================================================
SCRIPT_DIR  = Path(__file__).resolve().parent

RAW_DIR     = SCRIPT_DIR / "Raw Images"
ANNO_DIR    = SCRIPT_DIR / "Annotated Images"
WORK_DIR    = SCRIPT_DIR / "processed_dataset"
RUNS_DIR    = SCRIPT_DIR / "runs"

IMG_SIZE    = 832
SPLIT_RATIO = 0.85
RANDOM_SEED = 42
EPOCHS      = 100
BATCH       = 8

CLASS_MAP = {
    "car":0,        "Car":0,
    "cng":0,        "Cng":0,        "CNG":0,
    "truck":0,      "Truck":0,
    "mini-truck":0, "Mini-Truck":0,
    "pickup":0,     "Pickup":0,
    "microbus":0,   "Microbus":0,
    "ambulance":0,  "Ambulance":0,
    "bus":1,        "Bus":1,
    "minibus":1,    "Minibus":1,
    "bike":2,       "Bike":2,
    "motorcycle":2, "Motorcycle":2,
    "rickshaw":2,   "Rickshaw":2,
    "bicycle":2,    "Bicycle":2,
    "cycle":2,      "Cycle":2,
    "person":3,     "Person":3,
    "people":3,     "People":3,
    "pedestrian":3, "Pedestrian":3,
}
CLASS_NAMES = ["vehicle", "bus", "bike", "person"]
id2name     = {0:"vehicle", 1:"bus", 2:"bike", 3:"person"}

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
#  GPU CHECK
# ============================================================
print(f"ultralytics : {ultralytics.__version__}")
print(f"torch       : {torch.__version__}")
print(f"CUDA        : {torch.version.cuda}")

if torch.cuda.is_available():
    DEVICE = 0
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"sm_cap      : {torch.cuda.get_device_capability(0)}")
    try:
        _ = torch.zeros(1).cuda()
        print("GPU test    : ✅ WORKING")
    except Exception as e:
        raise RuntimeError(f"GPU broken: {e}")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"
    print("⚠️  No CUDA GPU found — running on CPU (training will be slow).")

# ============================================================
#  SANITY CHECK
# ============================================================
for d, label in [(RAW_DIR, "Raw Images"), (ANNO_DIR, "Annotated Images")]:
    if not d.is_dir():
        raise RuntimeError(
            f"Folder not found: {d}\n"
            f"Make sure '{label}' sits in the same directory as yolov8m.py"
        )

# ============================================================
#  CREATE DIRECTORY STRUCTURE
# ============================================================
for sp in ("train", "valid"):
    (WORK_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("  BTF YOLOv8m PIPELINE — PHASE 1")
print("=" * 55)
print(f"  Model  : yolov8m.pt | Epochs: {EPOCHS} | Batch: {BATCH}")
print(f"  ImgSize: {IMG_SIZE}  | Device: {DEVICE}")

# ============================================================
#  STEP 1 — Collect image + XML pairs
#
#  The dataset has a mirrored folder tree:
#    Raw Images/Location X/Lane Type/Session/  → images
#    Annotated Images/Location X/Lane Type/Session/  → XMLs
#
#  Strategy:
#  1. Walk Annotated Images for every XML.
#  2. Compute its relative path inside Annotated Images.
#  3. Look in the same relative sub-folder inside Raw Images
#     for an image with the same stem.
#  4. Fallback: scan all of Raw Images by stem if not found
#     at the mirrored path (handles any naming inconsistency).
# ============================================================
print("\n[1/6] Collecting image+XML pairs ...")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Build a full stem → path index for every image in Raw Images
raw_img_index = {}
for p in RAW_DIR.rglob("*"):
    if p.suffix.lower() in IMG_EXTS:
        raw_img_index[p.stem.lower()] = p

# Collect all XMLs from Annotated Images
all_xmls = [p for p in ANNO_DIR.rglob("*") if p.suffix.lower() == ".xml"]

matched = []
unmatched_xmls = 0

for xml_p in all_xmls:
    img_stem = xml_p.stem.lower()

    # --- Primary: mirror the relative sub-path into Raw Images ---
    rel = xml_p.relative_to(ANNO_DIR)          # e.g. Location 1/Double Lane/session/foo.xml
    mirror_dir = RAW_DIR / rel.parent           # Raw Images/Location 1/Double Lane/session/
    img_p = None

    if mirror_dir.is_dir():
        for ext in IMG_EXTS:
            candidate = mirror_dir / (xml_p.stem + ext)
            if candidate.exists():
                img_p = candidate
                break
        # Also try case-insensitive match within that folder
        if img_p is None:
            for f in mirror_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS and f.stem.lower() == img_stem:
                    img_p = f
                    break

    # --- Fallback: search entire Raw Images by stem ---
    if img_p is None:
        img_p = raw_img_index.get(img_stem)

    if img_p is not None:
        matched.append((str(img_p), str(xml_p)))
    else:
        unmatched_xmls += 1

print(f"  Images in Raw Images     : {len(raw_img_index)}")
print(f"  XMLs in Annotated Images : {len(all_xmls)}")
print(f"  Matched pairs            : {len(matched)}")
if unmatched_xmls:
    print(f"  ⚠️  XMLs with no matching image: {unmatched_xmls}")

if not matched:
    sample_imgs = [p.name for p in list(raw_img_index.values())[:5]]
    sample_xmls = [p.name for p in all_xmls[:5]]
    raise RuntimeError(
        "No image+XML pairs could be matched.\n\n"
        f"  Sample image names : {sample_imgs}\n"
        f"  Sample XML names   : {sample_xmls}\n\n"
        "  Expected: Raw Images and Annotated Images share the same\n"
        "  subfolder structure with matching filenames at each level."
    )

# ============================================================
#  STEP 2 — Remove duplicates
# ============================================================
print("\n[2/6] Removing duplicates ...")
seen, unique, dups = set(), [], 0
for img_p, xml_p in tqdm(matched, desc="  hashing"):
    try:
        h = hashlib.md5(open(img_p, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append((img_p, xml_p))
        else:
            dups += 1
    except Exception:
        pass
print(f"  Duplicates: {dups} | Unique: {len(unique)} ✅")

# ============================================================
#  STEP 3 — XML → YOLO conversion
# ============================================================
unrecognized = collections.Counter()

def xml_to_yolo(xml_path):
    lines = []
    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    size  = root.find("size")
    if size is None: return []
    W = int(float(size.find("width").text))
    H = int(float(size.find("height").text))
    if W <= 0 or H <= 0: return []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text: continue
        raw = name_el.text.strip()
        cls = CLASS_MAP.get(raw, CLASS_MAP.get(raw.lower()))
        if cls is None:
            unrecognized[raw] += 1; continue
        bb = obj.find("bndbox")
        if bb is None: continue
        xmin=float(bb.find("xmin").text); ymin=float(bb.find("ymin").text)
        xmax=float(bb.find("xmax").text); ymax=float(bb.find("ymax").text)
        xmin,xmax=min(xmin,xmax),max(xmin,xmax)
        ymin,ymax=min(ymin,ymax),max(ymin,ymax)
        if xmax<=xmin or ymax<=ymin: continue
        cx=min(max(((xmin+xmax)/2)/W,0.001),0.999)
        cy=min(max(((ymin+ymax)/2)/H,0.001),0.999)
        bw=min(max((xmax-xmin)/W,0.001),0.999)
        bh=min(max((ymax-ymin)/H,0.001),0.999)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

# ============================================================
#  STEP 4 — Build dataset + oversample bike/bus
# ============================================================
print("\n[3/6] Building YOLO dataset ...")
random.shuffle(unique)
cut    = int(len(unique) * SPLIT_RATIO)
splits = {"train": unique[:cut], "valid": unique[cut:]}
skipped = 0
label_counts  = {"train": collections.Counter(), "valid": collections.Counter()}
converted_train = []

for sp, dataset in splits.items():
    for img_path, xml_path in tqdm(dataset, desc=f"  {sp}"):
        try:
            lines = xml_to_yolo(xml_path)
            if not lines: skipped += 1; continue
            img  = Image.open(img_path).convert("RGB")
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).save(
                str(WORK_DIR / "images" / sp / (stem + ".jpg")), "JPEG", quality=95)
            with open(WORK_DIR / "labels" / sp / (stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            cls_set = {int(l.split()[0]) for l in lines}
            for l in lines: label_counts[sp][int(l.split()[0])] += 1
            if sp == "train":
                converted_train.append((stem, cls_set))
        except Exception:
            skipped += 1

# Oversample minority classes: bike / bus
os_count = 0
for stem, cls_set in converted_train:
    if {1, 2} & cls_set:
        si = WORK_DIR / "images" / "train" / (stem + ".jpg")
        sl = WORK_DIR / "labels" / "train" / (stem + ".txt")
        for i in range(1, 3):
            di = WORK_DIR / "images" / "train" / f"{stem}_os{i}.jpg"
            dl = WORK_DIR / "labels" / "train" / f"{stem}_os{i}.txt"
            if not di.exists():
                shutil.copy2(si, di); shutil.copy2(sl, dl); os_count += 1

train_n = len(list((WORK_DIR / "images" / "train").iterdir()))
valid_n = len(list((WORK_DIR / "images" / "valid").iterdir()))
print(f"\n  Train: {train_n} | Valid: {valid_n} | Skipped: {skipped}")
print(f"  Oversampled (bike/bus): {os_count} extra copies")

print("\n  📊 Class distribution:")
for sp in ("train", "valid"):
    dist = " | ".join(f"{id2name.get(c,c)}: {n}" for c,n in sorted(label_counts[sp].items()))
    print(f"  [{sp}] {dist}")

if unrecognized:
    print(f"\n  ⚠️  Unrecognized labels: {dict(unrecognized)}")
else:
    print("\n  ✅ All labels recognized")

if train_n == 0:
    raise RuntimeError("Training set is empty — check your data folders.")

# ============================================================
#  STEP 5 — data.yaml
# ============================================================
print("\n[4/6] Writing data.yaml ...")
yaml_path = WORK_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {WORK_DIR.as_posix()}\ntrain: images/train\nval: images/valid\nnc: 4\nnames: {CLASS_NAMES}\n")
print(f"  ✅ {yaml_path}")

# ============================================================
#  STEP 6 — Load YOLOv8m
# ============================================================
print("\n[5/6] Loading YOLOv8m ...")
model = YOLO("yolov8m.pt")
print("  ✅ yolov8m loaded")

# ============================================================
#  TRAIN
# ============================================================
print("\n[6/6] Training ...")
print("=" * 55)

model.train(
    data            = str(yaml_path),
    epochs          = EPOCHS,
    imgsz           = IMG_SIZE,
    batch           = BATCH,
    name            = "BTF_YOLOv8m",
    project         = str(RUNS_DIR),
    device          = DEVICE,
    patience        = 20,
    save            = True,
    plots           = True,
    workers         = 2,
    exist_ok        = True,
    optimizer       = "auto",
    lr0             = 0.01,
    lrf             = 0.005,
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 5.0,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    box             = 7.5,
    cls             = 0.5,
    dfl             = 1.5,
    mosaic          = 1.0,
    mixup           = 0.2,
    copy_paste      = 0.0,
    degrees         = 10.0,
    translate       = 0.15,
    scale           = 0.6,
    shear           = 2.0,
    perspective     = 0.0001,
    flipud          = 0.01,
    fliplr          = 0.5,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    close_mosaic    = 15,
    nbs             = 64,
    amp             = False,
)

best_pt = RUNS_DIR / "BTF_YOLOv8m" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = RUNS_DIR / "BTF_YOLOv8m" / "weights" / "last.pt"
print(f"\n✅ Training complete: {best_pt}")

# ============================================================
#  EVALUATE
# ============================================================
print("\n" + "=" * 55)
print("  ACCURACY EVALUATION")
print("=" * 55)

model   = YOLO(str(best_pt))
metrics = model.val(
    data=str(yaml_path), imgsz=IMG_SIZE, batch=BATCH,
    conf=0.001, iou=0.50, device=DEVICE,
    plots=True, save_json=True)

P    = float(metrics.box.mp    if metrics.box.mp    else 0)
R    = float(metrics.box.mr    if metrics.box.mr    else 0)
m50  = float(metrics.box.map50 if metrics.box.map50 else 0)
m595 = float(metrics.box.map   if metrics.box.map   else 0)
F1   = 2 * P * R / (P + R + 1e-9)

print("\n" + "=" * 55)
print("  FINAL RESULTS — YOLOv8m Phase 1")
print("=" * 55)
print(f"  Precision          : {P:.4f}  ({P*100:.2f}%)")
print(f"  Recall             : {R:.4f}  ({R*100:.2f}%)")
print(f"  F1 Score           : {F1:.4f}  ({F1*100:.2f}%)")
print(f"  mAP @ IoU=0.50     : {m50:.4f}  ({m50*100:.2f}%)")
print(f"  mAP @ IoU=0.50:0.95: {m595:.4f}  ({m595*100:.2f}%)")
print("=" * 55)

print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
print("  " + "-" * 55)

for i in range(len(CLASS_NAMES)):
    n = CLASS_NAMES[i]
    try:   p_c = float(metrics.box.p[i])
    except: p_c = 0
    try:   r_c = float(metrics.box.r[i])
    except: r_c = 0
    try:   a50 = float(metrics.box.ap50[i])
    except: a50 = 0
    try:   ap  = float(metrics.box.ap[i])
    except: ap  = 0
    print(f"  {n:<12} {p_c:>10.4f} {r_c:>10.4f} {a50:>10.4f} {ap:>10.4f}")

print("  " + "-" * 55)
print(f"  {'MEAN':<12} {P:>10.4f} {R:>10.4f} {m50:>10.4f} {m595:>10.4f}")
print("=" * 55)

# ============================================================
#  PLOTS
# ============================================================
pngs = sorted(glob.glob(str(RUNS_DIR / "BTF_YOLOv8m" / "*.png")))
if pngs:
    cols = 2; rows = (len(pngs) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 7))
    axes = axes.flatten()
    for ax, p in zip(axes, pngs):
        try: ax.imshow(mpimg.imread(p)); ax.set_title(os.path.basename(p), fontsize=11)
        except: pass
        ax.axis("off")
    for ax in axes[len(pngs):]: ax.axis("off")
    plt.suptitle("YOLOv8m — Phase 1 Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(SCRIPT_DIR / "Phase1_plots.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("\n✅ Plots saved → Phase1_plots.png")

val_imgs = glob.glob(str(WORK_DIR / "images" / "valid" / "*.jpg"))
if val_imgs:
    sample = random.sample(val_imgs, min(6, len(val_imgs)))
    model.predict(source=sample, imgsz=IMG_SIZE, conf=0.25, device=DEVICE,
                  save=True, project=str(RUNS_DIR / "predictions"),
                  name="sample", exist_ok=True, verbose=False)
    preds = sorted(glob.glob(str(RUNS_DIR / "predictions" / "sample" / "*.jpg")))[:6]
    if preds:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, p in zip(axes.flatten(), preds):
            try: ax.imshow(mpimg.imread(p)); ax.set_title(os.path.basename(p), fontsize=8)
            except: pass
            ax.axis("off")
        plt.suptitle("YOLOv8m Phase 1 — Sample Predictions", fontsize=13)
        plt.tight_layout()
        plt.savefig(str(SCRIPT_DIR / "Phase1_predictions.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("✅ Predictions saved → Phase1_predictions.png")

shutil.copy(str(best_pt), str(SCRIPT_DIR / "BTF_YOLOv8m_best.pt"))
print(f"\n✅ best.pt → BTF_YOLOv8m_best.pt")
print("\n🏁 Phase 1 complete!")
print("   Share the per-class table → Phase 2 will fix weak classes.")