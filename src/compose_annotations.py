"""Compose final box annotations from OWLv2 proposals + rules + manual overrides
(manual overrides were set after visual review of overlay sheets)."""
import json
from common import image_meta, ROOT

p1 = json.load(open(ROOT / "data" / "owl_proposals.json"))
p2 = json.load(open(ROOT / "data" / "owl_proposals2.json"))

def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
def area(a):
    return (a[2] - a[0]) * (a[3] - a[1])

# Manual boxes in pixel coords (images are 3024x3024), fractions * 3024. Filled after review.
MANUAL = json.load(open(ROOT / "data" / "manual_boxes.json")) if (ROOT / "data" / "manual_boxes.json").exists() else {}

def saturation_box(path, thr=0.28):
    """Studio shots: grey background, toy+teal base are saturated -> bbox of largest saturated blob."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    im = Image.open(path).convert("HSV"); W = im.size[0]
    small = np.asarray(im.resize((504, 504))).astype(float) / 255
    mask = (small[..., 1] > thr) & (small[..., 2] > 0.15)
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, [k + 1 for k in range(n) if sizes[k] > 0.15 * sizes.max()])
    ys, xs = np.where(keep); s = W / 504
    return [xs.min() * s, ys.min() * s, (xs.max() + 1) * s, (ys.max() + 1) * s]

ann = {}
for m in image_meta():
    i = m["id"]; W = p1[i]["W"]; H = p1[i]["H"]
    boxes = {b["cls"]: b["box"] for b in p1[i]["boxes"]}
    src = {c: "owlv2" for c in boxes}
    if m["label"] == "penguin":
        cand = p2[i]["a white toy figurine"]
        if cand and cand[0]["score"] > 0.3:
            boxes["penguin"] = cand[0]["box"]; src["penguin"] = "owlv2-alt-prompt"
    if "hippo" in boxes:
        hb = boxes["hippo"]
        puz = [d for d in p2[i]["a teal puzzle piece"] if d["score"] > 0.04 and inter(d["box"], hb) > 0.2 * area(hb)
               and area(d["box"]) < 6 * area(hb)]
        if puz:
            boxes["hippo"] = union(hb, puz[0]["box"]); src["hippo"] = "owlv2+puzzle-union"
    if m["label"] == "hippo" and m["domain"] == "studio":
        sb = saturation_box(m["path"])
        if sb is not None:
            boxes["hippo"] = sb; src["hippo"] = "saturation-mask"
    for c, b in MANUAL.get(i, {}).items():
        boxes[c] = [v * W for v in b]; src[c] = "manual"
    ann[i] = dict(W=W, H=H, label=m["label"], domain=m["domain"],
                  boxes=[dict(cls=c, box=[round(v, 1) for v in b], source=src[c]) for c, b in boxes.items()])
json.dump(ann, open(ROOT / "data" / "annotations.json", "w"), indent=1)
from collections import Counter
print(Counter(b["source"] for a in ann.values() for b in a["boxes"]))
