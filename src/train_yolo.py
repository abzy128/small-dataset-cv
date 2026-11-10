"""Train an Ultralytics detector on a split, predict on test images, evaluate with the common evaluator.
usage: train_yolo.py <run_name> <split> <model(.pt|.yaml)> [epochs] [imgsz] [extra_train_dir]"""
import json, os, sys, time, shutil
from pathlib import Path
from ultralytics import YOLO, RTDETR
from common import CLASSES, ROOT
from det_eval import eval_split, test_ids

name, split, model = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 640
extra = sys.argv[6] if len(sys.argv) > 6 else None
data = ROOT / "data" / "yolo" / split / "data.yaml"
if extra and extra.startswith("list:"):  # train on a subset listed in a txt file
    y = data.read_text().replace("train: images/train", f"train: {extra[5:]}")
    data = ROOT / "data" / "yolo" / split / f"data_{name}.yaml"; data.write_text(y)
elif extra:  # add synthetic training images (list of dirs) via a derived yaml
    y = data.read_text().replace("train: images/train", f"train: [images/train, {extra}]")
    data = ROOT / "data" / "yolo" / split / f"data_{name}.yaml"; data.write_text(y)
W = ROOT / "weights"; W.mkdir(exist_ok=True)
mpath = str(W / model) if model.endswith(".pt") else model
Cls = RTDETR if "rtdetr" in model else YOLO
m = Cls(mpath)
t0 = time.time()
m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=8, device="mps", project=str(ROOT / "runs" / "det"), name=name,
        exist_ok=True, seed=int(os.environ.get("SEED", 0)), deterministic=False, workers=2, plots=True, verbose=False, patience=1000,
        close_mosaic=10, pretrained=model.endswith(".pt"), val=False)
train_time = time.time() - t0
last = ROOT / "runs" / "det" / name / "weights" / "last.pt"   # NOT best.pt -> no model selection on test
m = Cls(str(last))
preds = {}; t1 = time.time()
ids = test_ids(split)
for i in ids:
    r = m.predict(str(ROOT / "data" / "img1024" / f"{i}.jpg"), imgsz=imgsz, conf=0.001, iou=0.6, device="mps", verbose=False, max_det=100)[0]
    preds[i] = [dict(cls=CLASSES[int(c)], score=float(s), box=[float(v) for v in b])
                for b, s, c in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu())]
infer_ms = (time.time() - t1) / len(ids) * 1000
res = eval_split(preds, split)
res.update(train_time_s=train_time, infer_ms=infer_ms, epochs=epochs, imgsz=imgsz, model=model, params=sum(p.numel() for p in m.model.parameters()))
out = ROOT / "results" / "det"; out.mkdir(parents=True, exist_ok=True)
json.dump(preds, open(out / f"preds_{name}.json", "w"))
json.dump(res, open(out / f"metrics_{name}.json", "w"), indent=1)
print(f"{name}: all mAP50={res['all']['mAP50']:.3f} mAP50-95={res['all']['mAP50_95']:.3f} | single {res['single']['mAP50']:.3f} | multi {res['multi']['mAP50']:.3f} | {train_time:.0f}s")
