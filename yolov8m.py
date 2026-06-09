# ============================================================
#  BTF YOLOv8m PHASE 1
#  VS CODE / LOCAL MACHINE VERSION
#  Compatible: torch>=2.0 | ultralytics latest
#  GPU: Any CUDA-capable GPU (tested on Tesla P100, RTX series)
# ============================================================
#
#  FOLDER STRUCTURE EXPECTED:
#  project_root/
#  ├── Annotated Images/    ← XML annotation files (Pascal VOC format)
#  ├── Raw Images/          ← Corresponding image files (.jpg/.jpeg/.png)
#  ├── yolov8m.py           ← this script
#  └── requirements.txt
#
#  SET ENVIRONMENT VARIABLES (optional overrides):
#    BTF_RAW_DIR        → path to raw images folder
#    BTF_ANNO_DIR       → path to annotated images (XMLs) folder
#    BTF_WORK_DIR       → where processed dataset is written
#    BTF_RUNS_DIR       → where training runs are saved
#    BTF_EPOCHS         → number of training epochs (default: 100)
#    BTF_BATCH          → batch size (default: 8)
#    BTF_IMG_SIZE       → input image size (default: 832)
#    BTF_DEVICE         → cuda device index or "cpu" (default: 0)
#    BTF_SPLIT_RATIO    → train/val split ratio (default: 0.85)
#
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
#  RESOLVE PATHS — script lives next to the two data folders
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

def _env(key, default):
    return os.environ.get(key, str(default))

def _resolve_data_dir(env_key: str, folder_name: str) -> Path:
    configured = os.environ.get(env_key)
    if configured:
        return Path(configured).expanduser().resolve()

    exact = SCRIPT_DIR / folder_name
    if exact.exists():
        return exact

    # Windows is usually case-insensitive, but this keeps the message nice and
    # also helps if the project is later run from a case-sensitive filesystem.
    wanted = folder_name.casefold()
    for child in SCRIPT_DIR.iterdir():
        if child.is_dir() and child.name.casefold() == wanted:
            return child
    return exact

RAW_DIR    = _resolve_data_dir("BTF_RAW_DIR", "Raw Images")
ANNO_DIR   = _resolve_data_dir("BTF_ANNO_DIR", "Annotated Images")
WORK_DIR   = Path(_env("BTF_WORK_DIR",  SCRIPT_DIR / "processed_dataset"))
RUNS_DIR   = Path(_env("BTF_RUNS_DIR",  SCRIPT_DIR / "runs"))
OUTPUT_DIR = SCRIPT_DIR  # Phase-level result files land here

IMG_SIZE     = int(_env("BTF_IMG_SIZE",    832))
SPLIT_RATIO  = float(_env("BTF_SPLIT_RATIO", 0.85))
RANDOM_SEED  = 42
EPOCHS       = int(_env("BTF_EPOCHS",  100))
BATCH        = int(_env("BTF_BATCH",   8))
DEVICE_CFG   = _env("BTF_DEVICE", "0")

# ============================================================
#  STARTUP — library versions & GPU check
# ============================================================
print(f"ultralytics : {ultralytics.__version__}")
print(f"torch       : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")

if DEVICE_CFG.lower() == "cpu":
    DEVICE = "cpu"
    print("Device      : CPU (forced via BTF_DEVICE=cpu)")
elif torch.cuda.is_available():
    DEVICE = int(DEVICE_CFG)
    print(f"GPU         : {torch.cuda.get_device_name(DEVICE)}")
    print(f"sm_cap      : {torch.cuda.get_device_capability(DEVICE)}")
    try:
        _ = torch.zeros(1).cuda()
        print("GPU test    : ✅ WORKING")
    except Exception as e:
        raise RuntimeError(f"GPU broken: {e}")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"
    print("⚠️  No CUDA GPU found — falling back to CPU (training will be slow).")

# ============================================================
#  CLASS MAP  (identical to original)
# ============================================================
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
#  CREATE DIRECTORY STRUCTURE
# ============================================================
for sp in ("train", "valid"):
    (WORK_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("  BTF YOLOv8m PIPELINE — PHASE 1")
print("=" * 55)
print(f"  Raw images  : {RAW_DIR}")
print(f"  Annotations : {ANNO_DIR}")
print(f"  Work dir    : {WORK_DIR}")
print(f"  Runs dir    : {RUNS_DIR}")
print(f"  Model       : yolov8m.pt | Epochs: {EPOCHS} | Batch: {BATCH}")
print(f"  ImgSize     : {IMG_SIZE}  | Device: {DEVICE}")

missing_data_dirs = []
if not RAW_DIR.is_dir():
    missing_data_dirs.append(f"Raw images folder not found: {RAW_DIR}")
if not ANNO_DIR.is_dir():
    missing_data_dirs.append(f"Annotation folder not found: {ANNO_DIR}")
if missing_data_dirs:
    raise RuntimeError(
        "Dataset folders are missing.\n"
        + "\n".join(f"  - {msg}" for msg in missing_data_dirs)
        + "\n\nExpected layout:\n"
        f"  {SCRIPT_DIR}\\Raw Images\\        -> .jpg/.jpeg/.png files\n"
        f"  {SCRIPT_DIR}\\Annotated Images\\  -> .xml files\n"
        "\nYou can also override paths with BTF_RAW_DIR and BTF_ANNO_DIR."
    )

# ============================================================
#  STEP 1 — Co-located image+XML pairs
#
#  Strategy (mirrors original):
#  • Walk ANNO_DIR for XML files.
#  • For each XML, look for an image with the same stem in RAW_DIR
#    (also walks subdirectories so nested structures are handled).
#  Images are read only from RAW_DIR and XML files only from ANNO_DIR.
# ============================================================
print("\n[1/6] Collecting image+XML pairs ...")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def _build_stem_index(root_dir: Path, exts: set) -> dict:
    """Return {stem_lower: absolute_path} for all files with given extensions."""
    index = {}
    for p in root_dir.rglob("*"):
        if p.suffix.lower() in exts:
            index[p.stem.lower()] = p
    return index

img_index  = _build_stem_index(RAW_DIR,  IMG_EXTS)
xml_index  = _build_stem_index(ANNO_DIR, {".xml"})

matched = []
used_xmls = set()
for stem, img_p in img_index.items():
    if stem in xml_index:
        xml_p = xml_index[stem]
        matched.append((str(img_p), str(xml_p)))
        used_xmls.add(xml_p)

# Some Pascal VOC datasets name XML files differently but store the real image
# filename inside <filename>. Fall back to that before giving up.
for xml_p in xml_index.values():
    if xml_p in used_xmls:
        continue
    try:
        root = ET.parse(xml_p).getroot()
        filename_el = root.find("filename")
        if filename_el is None or not filename_el.text:
            continue
        img_stem = Path(filename_el.text.strip()).stem.lower()
        img_p = img_index.get(img_stem)
        if img_p:
            matched.append((str(img_p), str(xml_p)))
            used_xmls.add(xml_p)
    except Exception:
        continue

print(f"  Matched pairs: {len(matched)}")
if not matched:
    raise RuntimeError(
        "No image+XML pairs found.\n"
        f"  Searched images in : {RAW_DIR}\n"
        f"  Searched XMLs in   : {ANNO_DIR}\n"
        "  Make sure XML <filename> values point to images in Raw Images, or use matching stems "
        "(e.g. img001.jpg / img001.xml)."
    )

# ============================================================
#  STEP 2 — Remove duplicates  (identical to original)
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
#  STEP 3 — XML → YOLO conversion  (identical to original)
# ============================================================
unrecognized = collections.Counter()

def xml_to_yolo(xml_path: str) -> list:
    lines = []
    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    size  = root.find("size")
    if size is None:
        return []
    W = int(float(size.find("width").text))
    H = int(float(size.find("height").text))
    if W <= 0 or H <= 0:
        return []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        raw = name_el.text.strip()
        cls = CLASS_MAP.get(raw, CLASS_MAP.get(raw.lower()))
        if cls is None:
            unrecognized[raw] += 1
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = min(max(((xmin + xmax) / 2) / W, 0.001), 0.999)
        cy = min(max(((ymin + ymax) / 2) / H, 0.001), 0.999)
        bw = min(max((xmax - xmin) / W,        0.001), 0.999)
        bh = min(max((ymax - ymin) / H,        0.001), 0.999)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

# ============================================================
#  STEP 4 — Build dataset + oversample bike/bus  (identical)
# ============================================================
print("\n[3/6] Building YOLO dataset ...")
random.shuffle(unique)
cut    = int(len(unique) * SPLIT_RATIO)
splits = {"train": unique[:cut], "valid": unique[cut:]}
skipped = 0
label_counts    = {"train": collections.Counter(), "valid": collections.Counter()}
converted_train = []

for sp, dataset in splits.items():
    for img_path, xml_path in tqdm(dataset, desc=f"  {sp}"):
        try:
            lines = xml_to_yolo(xml_path)
            if not lines:
                skipped += 1
                continue
            img  = Image.open(img_path).convert("RGB")
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).save(
                str(WORK_DIR / "images" / sp / (stem + ".jpg")),
                "JPEG", quality=95
            )
            with open(WORK_DIR / "labels" / sp / (stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            cls_set = {int(l.split()[0]) for l in lines}
            for l in lines:
                label_counts[sp][int(l.split()[0])] += 1
            if sp == "train":
                converted_train.append((stem, cls_set))
        except Exception:
            skipped += 1

# Oversample minority classes: bike / bus  (identical to original)
os_count = 0
for stem, cls_set in converted_train:
    if {1, 2} & cls_set:
        si = WORK_DIR / "images" / "train" / (stem + ".jpg")
        sl = WORK_DIR / "labels" / "train" / (stem + ".txt")
        for i in range(1, 3):
            di = WORK_DIR / "images" / "train" / f"{stem}_os{i}.jpg"
            dl = WORK_DIR / "labels" / "train" / f"{stem}_os{i}.txt"
            if not di.exists():
                shutil.copy2(si, di)
                shutil.copy2(sl, dl)
                os_count += 1

train_n = len(list((WORK_DIR / "images" / "train").iterdir()))
valid_n = len(list((WORK_DIR / "images" / "valid").iterdir()))
print(f"\n  Train: {train_n} | Valid: {valid_n} | Skipped: {skipped}")
print(f"  Oversampled (bike/bus): {os_count} extra copies")

print("\n  📊 Class distribution:")
for sp in ("train", "valid"):
    dist = " | ".join(
        f"{id2name.get(c, c)}: {n}"
        for c, n in sorted(label_counts[sp].items())
    )
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
    f.write(
        f"path: {WORK_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/valid\n"
        f"nc: 4\n"
        f"names: {CLASS_NAMES}\n"
    )
print(f"  ✅ {yaml_path}")

# ============================================================
#  STEP 6 — Load YOLOv8m
# ============================================================
print("\n[5/6] Loading YOLOv8m ...")
model = YOLO("yolov8m.pt")
print("  ✅ yolov8m loaded")

# ============================================================
#  TRAIN  (all hyperparameters identical to original)
# ============================================================
print("\n[6/6] Training ...")
print("=" * 55)

model.train(
    data             = str(yaml_path),
    epochs           = EPOCHS,
    imgsz            = IMG_SIZE,
    batch            = BATCH,
    name             = "BTF_YOLOv8m",
    project          = str(RUNS_DIR),
    device           = DEVICE,
    patience         = 20,
    save             = True,
    plots            = True,
    workers          = 2,
    exist_ok         = True,
    optimizer        = "auto",
    lr0              = 0.01,
    lrf              = 0.005,
    momentum         = 0.937,
    weight_decay     = 0.0005,
    warmup_epochs    = 5.0,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.1,
    box              = 7.5,
    cls              = 0.5,
    dfl              = 1.5,
    mosaic           = 1.0,
    mixup            = 0.2,
    copy_paste       = 0.0,
    degrees          = 10.0,
    translate        = 0.15,
    scale            = 0.6,
    shear            = 2.0,
    perspective      = 0.0001,
    flipud           = 0.01,
    fliplr           = 0.5,
    hsv_h            = 0.015,
    hsv_s            = 0.7,
    hsv_v            = 0.4,
    close_mosaic     = 15,
    nbs              = 64,
    amp              = False,
)

best_pt = RUNS_DIR / "BTF_YOLOv8m" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = RUNS_DIR / "BTF_YOLOv8m" / "weights" / "last.pt"
print(f"\n✅ Training complete: {best_pt}")

# ============================================================
#  EVALUATE  (identical to original)
# ============================================================
print("\n" + "=" * 55)
print("  ACCURACY EVALUATION")
print("=" * 55)

model   = YOLO(str(best_pt))
metrics = model.val(
    data      = str(yaml_path),
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    conf      = 0.001,
    iou       = 0.50,
    device    = DEVICE,
    plots     = True,
    save_json = True,
)

P    = float(metrics.box.mp   if metrics.box.mp   else 0)
R    = float(metrics.box.mr   if metrics.box.mr   else 0)
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

for i, ci in enumerate(range(len(CLASS_NAMES))):
    n = CLASS_NAMES[ci]
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
#  PLOTS  (saved to project root alongside the script)
# ============================================================
pngs = sorted(glob.glob(str(RUNS_DIR / "BTF_YOLOv8m" / "*.png")))
if pngs:
    cols = 2
    rows = (len(pngs) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 7))
    axes = axes.flatten()
    for ax, p in zip(axes, pngs):
        try:
            ax.imshow(mpimg.imread(p))
            ax.set_title(os.path.basename(p), fontsize=11)
        except Exception:
            pass
        ax.axis("off")
    for ax in axes[len(pngs):]:
        ax.axis("off")
    plt.suptitle("YOLOv8m — Phase 1 Results", fontsize=14)
    plt.tight_layout()
    out_plots = OUTPUT_DIR / "Phase1_plots.png"
    plt.savefig(str(out_plots), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Plots saved → {out_plots}")

val_imgs = glob.glob(str(WORK_DIR / "images" / "valid" / "*.jpg"))
if val_imgs:
    sample = random.sample(val_imgs, min(6, len(val_imgs)))
    model.predict(
        source  = sample,
        imgsz   = IMG_SIZE,
        conf    = 0.25,
        device  = DEVICE,
        save    = True,
        project = str(RUNS_DIR / "predictions"),
        name    = "sample",
        exist_ok= True,
        verbose = False,
    )
    preds = sorted(
        glob.glob(str(RUNS_DIR / "predictions" / "sample" / "*.jpg"))
    )[:6]
    if preds:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, p in zip(axes.flatten(), preds):
            try:
                ax.imshow(mpimg.imread(p))
                ax.set_title(os.path.basename(p), fontsize=8)
            except Exception:
                pass
            ax.axis("off")
        plt.suptitle("YOLOv8m Phase 1 — Sample Predictions", fontsize=13)
        plt.tight_layout()
        out_preds = OUTPUT_DIR / "Phase1_predictions.png"
        plt.savefig(str(out_preds), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"✅ Predictions saved → {out_preds}")

# Copy best weights to project root for easy access
dest_weights = OUTPUT_DIR / "BTF_YOLOv8m_best.pt"
shutil.copy(str(best_pt), str(dest_weights))
print(f"\n✅ best.pt → {dest_weights}")
print("\n🏁 Phase 1 complete!")
print("   Share the per-class table → Phase 2 will fix weak classes.")
