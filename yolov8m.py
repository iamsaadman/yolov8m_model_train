# ============================================================
#  BTF YOLOv8m PHASE 1
#  FIXED FOR: torch==2.5.1+cu121 | ultralytics latest
#  GPU: Tesla P100-PCIE-16GB (sm_60)
# ============================================================

import os, random, hashlib, glob, shutil, collections
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import ultralytics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(f"ultralytics : {ultralytics.__version__}")

# ── Verify GPU works before doing anything ───────────────────
print(f"torch     : {torch.__version__}")
print(f"CUDA      : {torch.version.cuda}")
print(f"GPU       : {torch.cuda.get_device_name(0)}")
print(f"sm_cap    : {torch.cuda.get_device_capability(0)}")
try:
    _ = torch.zeros(1).cuda()
    print("GPU test  : ✅ WORKING")
except Exception as e:
    raise RuntimeError(f"GPU still broken: {e}\nDid you restart the kernel after Cell 1?")

torch.backends.cudnn.benchmark = True

# ============================================================
#  CONFIG
# ============================================================
KAGGLE_INPUT = "/kaggle/input"
WORK_DIR     = "/kaggle/working/processed_dataset"
RUNS_DIR     = "/kaggle/working/runs"
IMG_SIZE     = 832
SPLIT_RATIO  = 0.85
RANDOM_SEED  = 42
EPOCHS       = 100
BATCH        = 8
DEVICE       = 0   # GPU confirmed working above

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

for sp in ("train", "valid"):
    os.makedirs(os.path.join(WORK_DIR, "images", sp), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "labels", sp), exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

print("=" * 55)
print("  BTF YOLOv8m PIPELINE — PHASE 1")
print("=" * 55)
print(f"  Model  : yolov8m.pt | Epochs: {EPOCHS} | Batch: {BATCH}")
print(f"  ImgSize: {IMG_SIZE}  | Device: GPU")

# ============================================================
#  STEP 1 — Co-located image+XML pairs
# ============================================================
print("\n[1/6] Collecting co-located image+XML pairs ...")
matched = []
for dirpath, _, filenames in os.walk(KAGGLE_INPUT):
    imgs, xmls = {}, {}
    for fname in filenames:
        stem, ext = os.path.splitext(fname)
        if ext.lower() in (".jpg", ".jpeg", ".png"):
            imgs[stem] = os.path.join(dirpath, fname)
        elif ext.lower() == ".xml":
            xmls[stem] = os.path.join(dirpath, fname)
    for stem in imgs:
        if stem in xmls:
            matched.append((imgs[stem], xmls[stem]))
print(f"  Matched pairs: {len(matched)}")
if not matched:
    raise RuntimeError("No pairs found.")

# ============================================================
#  STEP 2 — Remove duplicates
# ============================================================
print("\n[2/6] Removing duplicates ...")
seen, unique, dups = set(), [], 0
for img_p, xml_p in tqdm(matched, desc="  hashing"):
    try:
        h = hashlib.md5(open(img_p, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h); unique.append((img_p, xml_p))
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
                os.path.join(WORK_DIR, "images", sp, stem+".jpg"), "JPEG", quality=95)
            with open(os.path.join(WORK_DIR, "labels", sp, stem+".txt"), "w") as f:
                f.write("\n".join(lines))
            cls_set = {int(l.split()[0]) for l in lines}
            for l in lines: label_counts[sp][int(l.split()[0])] += 1
            if sp == "train":
                converted_train.append((stem, cls_set))
        except Exception:
            skipped += 1

# Oversample minority classes: bike/bus
os_count = 0
for stem, cls_set in converted_train:
    if {1, 2} & cls_set:
        si = os.path.join(WORK_DIR, "images", "train", stem+".jpg")
        sl = os.path.join(WORK_DIR, "labels", "train", stem+".txt")
        for i in range(1, 3):
            di = os.path.join(WORK_DIR, "images", "train", f"{stem}_os{i}.jpg")
            dl = os.path.join(WORK_DIR, "labels", "train", f"{stem}_os{i}.txt")
            if not os.path.exists(di):
                shutil.copy2(si, di); shutil.copy2(sl, dl); os_count += 1

train_n = len(os.listdir(os.path.join(WORK_DIR, "images", "train")))
valid_n = len(os.listdir(os.path.join(WORK_DIR, "images", "valid")))
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
    raise RuntimeError("Training set empty.")

# ============================================================
#  STEP 5 — data.yaml
# ============================================================
print("\n[4/6] Writing data.yaml ...")
yaml_path = os.path.join(WORK_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {WORK_DIR}\ntrain: images/train\nval: images/valid\nnc: 4\nnames: {CLASS_NAMES}\n")
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
    data            = yaml_path,
    epochs          = EPOCHS,
    imgsz           = IMG_SIZE,
    batch           = BATCH,
    name            = "BTF_YOLOv8m",
    project         = RUNS_DIR,
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

best_pt = os.path.join(RUNS_DIR, "BTF_YOLOv8m", "weights", "best.pt")
if not os.path.exists(best_pt):
    best_pt = os.path.join(RUNS_DIR, "BTF_YOLOv8m", "weights", "last.pt")
print(f"\n✅ Training complete: {best_pt}")

# ============================================================
#  EVALUATE
# ============================================================
print("\n" + "=" * 55)
print("  ACCURACY EVALUATION")
print("=" * 55)

model   = YOLO(best_pt)
metrics = model.val(
    data=yaml_path, imgsz=IMG_SIZE, batch=BATCH,
    conf=0.001, iou=0.50, device=DEVICE,
    plots=True, save_json=True)

P    = float(metrics.box.mp if metrics.box.mp else 0)
R    = float(metrics.box.mr if metrics.box.mr else 0)
m50  = float(metrics.box.map50 if metrics.box.map50 else 0)
m595 = float(metrics.box.map if metrics.box.map else 0)

F1   = 2*P*R/(P+R+1e-9)

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

idx = list(range(len(CLASS_NAMES)))

for i, ci in enumerate(idx):
    n = CLASS_NAMES[ci]

    try:
        p_c = float(metrics.box.p[i])
    except:
        p_c = 0

    try:
        r_c = float(metrics.box.r[i])
    except:
        r_c = 0

    try:
        a50 = float(metrics.box.ap50[i])
    except:
        a50 = 0

    try:
        ap = float(metrics.box.ap[i])
    except:
        ap = 0

    print(f"  {n:<12} {p_c:>10.4f} {r_c:>10.4f} {a50:>10.4f} {ap:>10.4f}")

print("  " + "-" * 55)
print(f"  {'MEAN':<12} {P:>10.4f} {R:>10.4f} {m50:>10.4f} {m595:>10.4f}")
print("=" * 55)

# ============================================================
#  PLOTS
# ============================================================
pngs = sorted(glob.glob(os.path.join(RUNS_DIR, "BTF_YOLOv8m", "*.png")))
if pngs:
    cols = 2; rows = (len(pngs)+1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*7))
    axes = axes.flatten()
    for ax, p in zip(axes, pngs):
        try: ax.imshow(mpimg.imread(p)); ax.set_title(os.path.basename(p), fontsize=11)
        except: pass
        ax.axis("off")
    for ax in axes[len(pngs):]: ax.axis("off")
    plt.suptitle("YOLOv8m — Phase 1 Results", fontsize=14)
    plt.tight_layout()
    plt.savefig("/kaggle/working/Phase1_plots.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n✅ Plots saved.")

val_imgs = glob.glob(os.path.join(WORK_DIR, "images", "valid", "*.jpg"))
if val_imgs:
    sample = random.sample(val_imgs, min(6, len(val_imgs)))
    model.predict(source=sample, imgsz=IMG_SIZE, conf=0.25, device=DEVICE,
                  save=True, project=os.path.join(RUNS_DIR, "predictions"),
                  name="sample", exist_ok=True, verbose=False)
    preds = sorted(glob.glob(
        os.path.join(RUNS_DIR, "predictions", "sample", "*.jpg")))[:6]
    if preds:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, p in zip(axes.flatten(), preds):
            try: ax.imshow(mpimg.imread(p)); ax.set_title(os.path.basename(p), fontsize=8)
            except: pass
            ax.axis("off")
        plt.suptitle("YOLOv8m Phase 1 — Sample Predictions", fontsize=13)
        plt.tight_layout()
        plt.savefig("/kaggle/working/Phase1_predictions.png", dpi=100, bbox_inches="tight")
        plt.close()
        print("✅ Predictions saved.")

shutil.copy(best_pt, "/kaggle/working/BTF_YOLOv8m_best.pt")
print(f"\n✅ best.pt → /kaggle/working/BTF_YOLOv8m_best.pt")
print("   Download: right panel → Output → BTF_YOLOv8m_best.pt")
print("\n🏁 Phase 1 complete!")
print("   Share the per-class table → Phase 2 will fix weak classes.")