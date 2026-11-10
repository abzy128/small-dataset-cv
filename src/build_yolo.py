"""Build YOLO-format detection datasets for two splits.
  S1_studio2desk : train = 41 studio single-object images; test = 13 desk single + 4 multi-object scenes
  S2_mixed       : train = 41 studio + 8 desk (2/class); test = 5 remaining desk single + 4 multi-object scenes
Images are the 1024px copies. Labels from data/annotations.json."""
import json, shutil
from common import image_meta, CLASSES, ROOT

ann = json.load(open(ROOT / "data" / "annotations.json"))
DESK_TRAIN = {"IMG_0476", "IMG_0477", "IMG_0479", "IMG_0480", "IMG_0482", "IMG_0483", "IMG_0486", "IMG_0487"}

def split_of(m, split):
    if m["label"] == "multi": return "test"
    if m["domain"] == "studio": return "train"
    if split == "S2_mixed" and m["id"] in DESK_TRAIN: return "train"
    return "test"

def write(split):
    base = ROOT / "data" / "yolo" / split
    shutil.rmtree(base, ignore_errors=True)
    meta = {}
    for m in image_meta():
        s = split_of(m, split); meta[m["id"]] = s
        (base / "images" / s).mkdir(parents=True, exist_ok=True); (base / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / "img1024" / (m["id"] + ".jpg"), base / "images" / s)
        a = ann[m["id"]]; W, H = a["W"], a["H"]
        with open(base / "labels" / s / (m["id"] + ".txt"), "w") as f:
            for b in a["boxes"]:
                x0, y0, x1, y1 = b["box"]
                f.write(f"{CLASSES.index(b['cls'])} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")
    (base / "data.yaml").write_text(f"path: {base}\ntrain: images/train\nval: images/train  # monitoring only; test is never used for model selection\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    json.dump(meta, open(base / "split.json", "w"), indent=1)
    print(split, {k: list(meta.values()).count(k) for k in ("train", "test")})

if __name__ == "__main__":
    for s in ("S1_studio2desk", "S2_mixed"):
        write(s)
