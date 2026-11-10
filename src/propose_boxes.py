"""Propose bounding boxes with OWLv2 (zero-shot), one query per known class."""
import json, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from common import image_meta, CLASSES, PROMPTS, ROOT

dev = "mps"
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
texts = [PROMPTS[c] for c in CLASSES]
out = {}
for m in image_meta():
    im = Image.open(m["path"]).convert("RGB")
    W, H = im.size
    inp = proc(text=[texts], images=im, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model(**inp)
    # owlv2 pads to square; images are square already
    res = proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=torch.tensor([[max(W,H), max(W,H)]]).to(dev))[0]
    dets = []
    for s, l, b in zip(res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu()):
        dets.append(dict(cls=CLASSES[int(l)], score=float(s), box=[float(x) for x in b]))
    want = CLASSES if m["label"] == "multi" else [m["label"]]
    boxes = []
    for c in want:
        cand = sorted([d for d in dets if d["cls"] == c], key=lambda d: -d["score"])
        if cand:
            boxes.append(cand[0])
    out[m["id"]] = dict(W=W, H=H, boxes=boxes, top_any=sorted(dets, key=lambda d: -d["score"])[:8])
    print(m["id"], [(b["cls"], round(b["score"], 3)) for b in boxes])
json.dump(out, open(ROOT / "data" / "owl_proposals.json", "w"), indent=1)
