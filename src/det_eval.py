"""Common detection evaluator: COCO-style mAP via torchmetrics, on any subset of test images.
Predictions format: {img_id: [{"cls": str, "score": float, "box": [x0,y0,x1,y1] in 1024px coords}]}"""
import json, torch
from torchmetrics.detection import MeanAveragePrecision
from common import CLASSES, ROOT, image_meta

ANN = json.load(open(ROOT / "data" / "annotations.json"))
META = {m["id"]: m for m in image_meta()}

def gt(img_id):
    a = ANN[img_id]; s = 1024 / a["W"]
    return [dict(cls=b["cls"], box=[v * s for v in b["box"]]) for b in a["boxes"]]

def evaluate(preds, ids):
    mp = MeanAveragePrecision(iou_type="bbox", class_metrics=True, backend="faster_coco_eval" if False else "pycocotools")
    P, T = [], []
    for i in ids:
        p = preds.get(i, [])
        P.append(dict(boxes=torch.tensor([d["box"] for d in p], dtype=torch.float32).reshape(-1, 4),
                      scores=torch.tensor([d["score"] for d in p], dtype=torch.float32),
                      labels=torch.tensor([CLASSES.index(d["cls"]) for d in p], dtype=torch.int64)))
        g = gt(i)
        T.append(dict(boxes=torch.tensor([d["box"] for d in g], dtype=torch.float32).reshape(-1, 4),
                      labels=torch.tensor([CLASSES.index(d["cls"]) for d in g], dtype=torch.int64)))
    mp.update(P, T); r = mp.compute()
    out = dict(mAP50_95=float(r["map"]), mAP50=float(r["map_50"]), mAP75=float(r["map_75"]), mAR100=float(r["mar_100"]))
    cl = r["classes"].tolist(); cl = cl if isinstance(cl, list) else [cl]
    out["per_class_AP50_95"] = {CLASSES[c]: float(v) for c, v in zip(cl, r["map_per_class"].reshape(-1).tolist())}
    return out

def eval_split(preds, split):
    sp = json.load(open(ROOT / "data" / "yolo" / split / "split.json"))
    test = [i for i, s in sp.items() if s == "test"]
    single = [i for i in test if META[i]["label"] != "multi"]; multi = [i for i in test if META[i]["label"] == "multi"]
    return dict(all=evaluate(preds, test), single=evaluate(preds, single), multi=evaluate(preds, multi))

def test_ids(split):
    sp = json.load(open(ROOT / "data" / "yolo" / split / "split.json"))
    return [i for i, s in sp.items() if s == "test"]
