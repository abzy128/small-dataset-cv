"""Extract frozen embeddings for all images with several pretrained backbones (with hflip TTA views)."""
import numpy as np, torch, timm
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from common import image_meta, ROOT, CLASSES

dev = "mps"
metas = image_meta()
import json, sys
ims = [Image.open(ROOT / "data" / "img1024" / (m["id"] + ".jpg")).convert("RGB") for m in metas]
out = ROOT / "results" / "features"; out.mkdir(parents=True, exist_ok=True)
SUFFIX = ""
if len(sys.argv) > 1 and sys.argv[1] == "crop":
    # object-centric variant: square crop around the (first) GT box with 20% padding
    ann = json.load(open(ROOT / "data" / "annotations.json")); SUFFIX = "_crop"
    def crop(im, a):
        s = im.size[0] / a["W"]; x0, y0, x1, y1 = [v * s for v in a["boxes"][0]["box"]]
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.2
        return im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    ims = [crop(im, ann[m["id"]]) for im, m in zip(ims, metas)]

def timm_feats(name):
    m = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    with torch.no_grad():
        f = [m(torch.stack([tf(im), tf(im.transpose(Image.FLIP_LEFT_RIGHT))]).to(dev)).cpu().numpy() for im in ims]
    return np.stack(f)  # N x 2 x D

def hf_feats(name, kind):
    if kind == "clip":
        m = CLIPModel.from_pretrained(name).to(dev).eval(); p = CLIPProcessor.from_pretrained(name)
        fn = lambda x: m.get_image_features(**x)
    else:
        m = AutoModel.from_pretrained(name).to(dev).eval(); p = AutoProcessor.from_pretrained(name)
        if kind == "siglip":
            fn = lambda x: m.get_image_features(**x)
        else:  # dinov2: CLS + mean patch
            def fn(x):
                h = m(**x).last_hidden_state
                return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    res = []
    with torch.no_grad():
        for im in ims:
            x = p(images=[im, im.transpose(Image.FLIP_LEFT_RIGHT)], return_tensors="pt").to(dev)
            o = fn(x); o = o.pooler_output if hasattr(o, "pooler_output") else o
            res.append(o.float().cpu().numpy())
    return np.stack(res)

jobs = {
    "resnet50_in1k": lambda: timm_feats("resnet50.a1_in1k"),
    "efficientnet_b0_in1k": lambda: timm_feats("efficientnet_b0.ra_in1k"),
    "convnext_tiny_in22k": lambda: timm_feats("convnext_tiny.fb_in22k_ft_in1k"),
    "clip_vitb16": lambda: hf_feats("openai/clip-vit-base-patch16", "clip"),
    "siglip2_base": lambda: hf_feats("google/siglip2-base-patch16-224", "siglip"),
    "dinov2_small": lambda: hf_feats("facebook/dinov2-small", "dino"),
    "dinov2_base": lambda: hf_feats("facebook/dinov2-base", "dino"),
}
for k, fn in jobs.items():
    try:
        f = fn(); np.save(out / f"{k}{SUFFIX}.npy", f); print(k, f.shape, flush=True)
    except Exception as e:
        print("FAILED", k, repr(e)[:300], flush=True)
np.save(out / "ids.npy", np.array([m["id"] for m in metas]))
