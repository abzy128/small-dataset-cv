"""Copy-paste synthetic data: paste SAM-masked TRAIN objects (random scale/flip/rotation/colour) onto TRAIN backgrounds.
Only train-split images are used as sources and backgrounds, so no test information leaks."""
import json, sys, random, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from common import ROOT, CLASSES

def build(split, n_images=300, seed=0):
    rnd = random.Random(seed)
    ann = json.load(open(ROOT / "data" / "annotations.json"))
    sp = json.load(open(ROOT / "data" / "yolo" / split / "split.json"))
    train = [i for i, s in sp.items() if s == "train"]
    objs = []
    for i in train:
        im = Image.open(ROOT / "data" / "img1024" / f"{i}.jpg").convert("RGB"); s = 1024 / ann[i]["W"]
        for k, b in enumerate(ann[i]["boxes"]):
            mk = Image.open(ROOT / "data" / "masks" / f"{i}_{k}.png")
            x0, y0, x1, y1 = [int(v * s) for v in b["box"]]
            bb = mk.crop((x0, y0, x1, y1)).getbbox()
            if bb is None: continue
            bb = (bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0)
            rgba = im.crop(bb); rgba.putalpha(mk.crop(bb).filter(ImageFilter.GaussianBlur(1.2)))
            objs.append((b["cls"], rgba))
    out_i = ROOT / "data" / "yolo" / split / "synth" / "images"; out_l = ROOT / "data" / "yolo" / split / "synth" / "labels"
    out_i.mkdir(parents=True, exist_ok=True); out_l.mkdir(parents=True, exist_ok=True)
    for n in range(n_images):
        bg_id = rnd.choice(train)
        bg = Image.open(ROOT / "data" / "img1024" / f"{bg_id}.jpg").convert("RGB")
        s = 1024 / ann[bg_id]["W"]
        boxes = [(b["cls"], [v * s for v in b["box"]]) for b in ann[bg_id]["boxes"]]
        if rnd.random() < 0.5:  # random zoom-in on background (retry until no original object is cut by the crop)
            for _try in range(20):
                z = rnd.uniform(0.5, 0.9); cw = int(1024 * z); cx, cy = rnd.randint(0, 1024 - cw), rnd.randint(0, 1024 - cw)
                f = 1024 / cw; nb, ok = [], True
                for c, (x0, y0, x1, y1) in boxes:
                    X0, Y0, X1, Y1 = (x0 - cx) * f, (y0 - cy) * f, (x1 - cx) * f, (y1 - cy) * f
                    if X0 >= 0 and Y0 >= 0 and X1 <= 1024 and Y1 <= 1024: nb.append((c, [X0, Y0, X1, Y1]))
                    elif X1 > 0 and Y1 > 0 and X0 < 1024 and Y0 < 1024: ok = False
                if ok:
                    bg = bg.crop((cx, cy, cx + cw, cy + cw)).resize((1024, 1024)); boxes = nb; break
        for _ in range(rnd.randint(2, 5)):
            c, o = rnd.choice(objs)
            target = rnd.uniform(0.04, 0.45) * 1024  # longest side in px: covers tiny desk-scale and close-up scale
            sc = target / max(o.size); o2 = o.resize((max(4, int(o.width * sc)), max(4, int(o.height * sc))))
            if rnd.random() < 0.5: o2 = o2.transpose(Image.FLIP_LEFT_RIGHT)
            o2 = o2.rotate(rnd.uniform(-25, 25), expand=True, resample=Image.BICUBIC)
            rgb = ImageEnhance.Brightness(o2.convert("RGB")).enhance(rnd.uniform(0.7, 1.2)); rgb.putalpha(o2.getchannel("A")); o2 = rgb
            for _try in range(30):
                x, y = rnd.randint(0, 1024 - o2.width), rnd.randint(0, 1024 - o2.height)
                bb = o2.getchannel("A").getbbox(); nbx = [x + bb[0], y + bb[1], x + bb[2], y + bb[3]]
                if all(nbx[2] < b[0] or nbx[0] > b[2] or nbx[3] < b[1] or nbx[1] > b[3] for _, b in boxes):
                    bg.paste(o2, (x, y), o2); boxes.append((c, nbx)); break
        bg.save(out_i / f"synth_{n:04d}.jpg", quality=90)
        with open(out_l / f"synth_{n:04d}.txt", "w") as f:
            for c, (x0, y0, x1, y1) in boxes:
                f.write(f"{CLASSES.index(c)} {(x0+x1)/2048:.6f} {(y0+y1)/2048:.6f} {(x1-x0)/1024:.6f} {(y1-y0)/1024:.6f}\n")
    print(split, "objects:", len(objs), "synthetic images:", n_images)

if __name__ == "__main__":
    for s in ("S1_studio2desk", "S2_mixed"):
        build(s)
