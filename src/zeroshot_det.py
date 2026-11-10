"""Zero-shot open-vocabulary detection baselines (no training): OWLv2 and Grounding DINO.
NOTE: OWLv2 proposals seeded ~75% of the GT boxes (53/70) (after review/correction), so its score is optimistically biased."""
import json, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection
from common import ROOT, CLASSES, image_meta
from det_eval import eval_split, test_ids

dev = "mps"
ids = test_ids("S1_studio2desk")  # superset of S2 test
PROMPTS = {"penguin": "a penguin figurine", "hippo": "a blue hippo toy", "battery": "a battery", "controller": "a game controller"}
res_all = {}

def owl():
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    m = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    preds = {}
    for i in ids:
        im = Image.open(ROOT / "data" / "img1024" / f"{i}.jpg").convert("RGB")
        x = proc(text=[[PROMPTS[c] for c in CLASSES]], images=im, return_tensors="pt").to(dev)
        with torch.no_grad(): o = m(**x)
        r = proc.post_process_grounded_object_detection(o, threshold=0.01, target_sizes=torch.tensor([[1024, 1024]]).to(dev))[0]
        preds[i] = [dict(cls=CLASSES[int(l)], score=float(s), box=[float(v) for v in b]) for s, l, b in zip(r["scores"].cpu(), r["labels"].cpu(), r["boxes"].cpu())]
    return preds

def gdino():
    name = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(name); m = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(dev).eval()
    text = ". ".join(PROMPTS[c] for c in CLASSES) + "."
    preds = {}
    for i in ids:
        im = Image.open(ROOT / "data" / "img1024" / f"{i}.jpg").convert("RGB")
        x = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad(): o = m(**x)
        r = proc.post_process_grounded_object_detection(o, x.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[(1024, 1024)])[0]
        labs = r.get("text_labels", r.get("labels"))
        out = []
        for s, l, b in zip(r["scores"].cpu(), labs, r["boxes"].cpu()):
            c = next((c for c in CLASSES if any(w in l for w in PROMPTS[c].split()[1:])), None)  # map phrase -> class
            if c: out.append(dict(cls=c, score=float(s), box=[float(v) for v in b]))
        preds[i] = out
    return preds

for name, fn in [("zeroshot_owlv2", owl), ("zeroshot_gdino_tiny", gdino)]:
    p = fn(); json.dump(p, open(ROOT / "results" / "det" / f"preds_{name}.json", "w"))
    for split in ("S1_studio2desk", "S2_mixed"):
        r = eval_split(p, split); r.update(model=name, train_time_s=0)
        tag = ("S1_" if split.startswith("S1") else "S2_") + name
        json.dump(r, open(ROOT / "results" / "det" / f"metrics_{tag}.json", "w"), indent=1)
        print(tag, f"mAP50={r['all']['mAP50']:.3f} mAP50-95={r['all']['mAP50_95']:.3f} single={r['single']['mAP50']:.3f} multi={r['multi']['mAP50']:.3f}")
