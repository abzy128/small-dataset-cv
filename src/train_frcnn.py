"""Fine-tune torchvision Faster R-CNN (ResNet50-FPN v2, COCO-pretrained) on a split; evaluate with the common evaluator.
usage: train_frcnn.py <run_name> <split> [epochs] [use_synth 0/1]"""
import json, os, sys, time, glob, random, numpy as np, torch
from PIL import Image
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from common import ROOT, CLASSES
from det_eval import eval_split, test_ids

name, split = sys.argv[1], sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
synth = len(sys.argv) > 4 and sys.argv[4] == "1"
SEED = int(os.environ.get("SEED", 0)); dev = "mps"; torch.manual_seed(SEED); random.seed(SEED)
base = ROOT / "data" / "yolo" / split
items = sorted(glob.glob(str(base / "images" / "train" / "*.jpg")))
if synth: items += sorted(glob.glob(str(base / "synth" / "images" / "*.jpg")))

def load(path):
    im = Image.open(path).convert("RGB").resize((640, 640)); boxes, labels = [], []
    for l in open(path.replace("/images/", "/labels/").replace(".jpg", ".txt")):
        c, x, y, w, h = map(float, l.split())
        boxes.append([(x - w / 2) * 640, (y - h / 2) * 640, (x + w / 2) * 640, (y + h / 2) * 640]); labels.append(int(c) + 1)
    return im, boxes, labels
data = [load(p) for p in items]
aug = T.Compose([T.RandomHorizontalFlip(), T.RandomPhotometricDistort(p=0.5), T.RandomZoomOut(fill=128, side_range=(1, 2.5), p=0.4),
                 T.RandomIoUCrop() if False else T.Identity(), T.Resize((640, 640)), T.ToImage(), T.ToDtype(torch.float32, scale=True), T.SanitizeBoundingBoxes()])

m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, min_size=640, max_size=640)
m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
m.to(dev)
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
bs = 4; steps = epochs * int(np.ceil(len(data) / bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=steps, pct_start=0.1)
t0 = time.time(); log = []
for e in range(epochs):
    m.train(); random.shuffle(data); tot = 0
    for b in range(0, len(data), bs):
        ims, tg = [], []
        for im, bx, lb in data[b:b + bs]:
            x, t = aug(im, {"boxes": tv_tensors.BoundingBoxes(bx, format="XYXY", canvas_size=(640, 640)), "labels": torch.tensor(lb)})
            ims.append(x.to(dev)); tg.append({"boxes": t["boxes"].as_subclass(torch.Tensor).float().to(dev), "labels": t["labels"].to(dev)})
        loss = sum(m(ims, tg).values()); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    log.append(tot); print(f"epoch {e+1} loss {tot:.3f}", flush=True)
train_time = time.time() - t0
m.eval(); preds = {}; ids = test_ids(split); t1 = time.time()
for i in ids:
    x = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(Image.open(ROOT / "data" / "img1024" / f"{i}.jpg").convert("RGB").resize((640, 640)))
    with torch.no_grad(): o = m([x.to(dev)])[0]
    preds[i] = [dict(cls=CLASSES[int(l) - 1], score=float(s), box=[float(v) * 1024 / 640 for v in b]) for b, s, l in zip(o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu())]
infer_ms = (time.time() - t1) / len(ids) * 1000
res = eval_split(preds, split)
res.update(train_time_s=train_time, infer_ms=infer_ms, epochs=epochs, imgsz=640, model="fasterrcnn_resnet50_fpn_v2", loss_log=log,
           params=sum(p.numel() for p in m.parameters()))
json.dump(preds, open(ROOT / "results" / "det" / f"preds_{name}.json", "w"))
json.dump(res, open(ROOT / "results" / "det" / f"metrics_{name}.json", "w"), indent=1)
print(f"{name}: all mAP50={res['all']['mAP50']:.3f} mAP50-95={res['all']['mAP50_95']:.3f} | single {res['single']['mAP50']:.3f} | multi {res['multi']['mAP50']:.3f} | {train_time:.0f}s")
