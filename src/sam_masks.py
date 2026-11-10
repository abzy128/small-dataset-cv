"""Box-prompted SAM masks for every annotated object (used for copy-paste synthesis and figures)."""
import json, numpy as np, torch
from PIL import Image
from transformers import SamModel, SamProcessor
from common import ROOT, image_meta

dev = "mps"
proc = SamProcessor.from_pretrained("facebook/sam-vit-base"); model = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
ann = json.load(open(ROOT / "data" / "annotations.json"))
out = ROOT / "data" / "masks"; out.mkdir(exist_ok=True)
stats = {}
for m in image_meta():
    im = Image.open(ROOT / "data" / "img1024" / f"{m['id']}.jpg").convert("RGB"); a = ann[m["id"]]; s = 1024 / a["W"]
    boxes = [[[v * s for v in b["box"]] for b in a["boxes"]]]
    inp = proc(im, input_boxes=boxes, return_tensors="pt")
    inp = {k: (v.float() if v.dtype == torch.float64 else v) for k, v in inp.items()}
    inp_dev = {k: v.to(dev) for k, v in inp.items() if k in ("pixel_values", "input_boxes")}
    with torch.no_grad():
        o = model(**inp_dev, multimask_output=True)
    masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"], inp["reshaped_input_sizes"])[0]
    iou = o.iou_scores.cpu()[0]
    for k, b in enumerate(a["boxes"]):
        j = int(iou[k].argmax()); mk = masks[k, j].numpy()
        Image.fromarray((mk * 255).astype(np.uint8)).save(out / f"{m['id']}_{k}.png")
        x0, y0, x1, y1 = [v * s for v in b["box"]]
        fill = mk[int(y0):int(y1), int(x0):int(x1)].mean()
        stats[f"{m['id']}_{k}"] = dict(cls=b["cls"], pred_iou=float(iou[k, j]), box_fill=float(fill))
json.dump(stats, open(out / "stats.json", "w"), indent=1)
print(len(stats), "masks; mean pred_iou", np.mean([v["pred_iou"] for v in stats.values()]))
