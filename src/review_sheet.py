"""Draw boxes from an annotation json onto contact sheets for visual review."""
import json, sys
from PIL import Image, ImageDraw
from common import image_meta, ROOT
COL = dict(penguin="red", hippo="blue", battery="orange", controller="magenta")
ann = json.load(open(sys.argv[1])); outp = sys.argv[2]; ids = sys.argv[3].split(",") if len(sys.argv) > 3 else None
metas = [m for m in image_meta() if ids is None or m["id"] in ids]
th, cols = 480, 5
rows = (len(metas) + cols - 1) // cols
sheet = Image.new("RGB", (cols * th, rows * th), "white")
for k, m in enumerate(metas):
    im = Image.open(m["path"]).convert("RGB"); W = im.size[0]
    im = im.resize((th, th)); s = th / W
    d = ImageDraw.Draw(im)
    for b in ann[m["id"]]["boxes"]:
        x0, y0, x1, y1 = [v * s for v in b["box"]]
        d.rectangle([x0, y0, x1, y1], outline=COL[b["cls"]], width=4)
    d.rectangle([0, 0, 110, 20], fill="black"); d.text((4, 4), m["id"][4:], fill="yellow")
    # grid ticks every 10% for coordinate reading
    for g in range(1, 10):
        d.line([g * th / 10, 0, g * th / 10, 6], fill="black"); d.line([0, g * th / 10, 6, g * th / 10], fill="black")
    sheet.paste(im, ((k % cols) * th, (k // cols) * th))
sheet.save(outp, quality=85)
