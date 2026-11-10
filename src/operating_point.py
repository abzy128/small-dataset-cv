"""Precision / recall / F1 at a fixed confidence threshold (deployment view), IoU>=0.5 greedy matching, per class-agnostic count."""
import json, glob, numpy as np
from pathlib import Path
from common import ROOT
from det_eval import gt, test_ids

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def op(preds, ids, thr):
    tp = fp = fn = 0
    for i in ids:
        g = gt(i); used = [False] * len(g)
        for d in sorted([d for d in preds.get(i, []) if d["score"] >= thr], key=lambda d: -d["score"]):
            best, bj = 0.5, -1
            for j, gg in enumerate(g):
                if not used[j] and gg["cls"] == d["cls"] and iou(d["box"], gg["box"]) >= best: best, bj = iou(d["box"], gg["box"]), j
            if bj >= 0: used[bj] = True; tp += 1
            else: fp += 1
        fn += used.count(False)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))

out = {}
for f in sorted(glob.glob(str(ROOT / "results" / "det" / "preds_*.json"))):
    name = Path(f).stem.replace("preds_", "")
    preds = json.load(open(f))
    splits = ["S1_studio2desk", "S2_mixed"] if name.startswith("zeroshot") else (["S1_studio2desk"] if name.startswith("S1") else ["S2_mixed"])
    for sp in splits:
        tag = name if not name.startswith("zeroshot") else sp[:2] + "_" + name
        out[tag] = {str(t): op(preds, test_ids(sp), t) for t in (0.25, 0.5)}
json.dump(out, open(ROOT / "results" / "det_operating_point.json", "w"), indent=1)
for k, v in out.items():
    o = v["0.5"]; print(f"{k:28s} P={o['precision']:.2f} R={o['recall']:.2f} F1={o['f1']:.2f} tp={o['tp']} fp={o['fp']} fn={o['fn']}")
