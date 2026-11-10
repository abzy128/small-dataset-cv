"""Second OWLv2 pass: extra prompts (penguin alternatives, puzzle-piece base) -> top detections per prompt."""
import json, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from common import image_meta, ROOT

dev = "mps"
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
texts = ["a white toy figurine", "a white penguin toy", "a teal puzzle piece", "a small toy"]
out = {}
for m in image_meta():
    im = Image.open(m["path"]).convert("RGB"); W, H = im.size
    inp = proc(text=[texts], images=im, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model(**inp)
    res = proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=torch.tensor([[W, H]]).to(dev))[0]
    per = {t: [] for t in texts}
    for s, l, b in zip(res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu()):
        per[texts[int(l)]].append(dict(score=float(s), box=[float(x) for x in b]))
    out[m["id"]] = {t: sorted(v, key=lambda d: -d["score"])[:5] for t, v in per.items()}
    print(m["id"], {t[:12]: round(v[0]["score"], 2) for t, v in out[m["id"]].items() if v})
json.dump(out, open(ROOT / "data" / "owl_proposals2.json", "w"), indent=1)
