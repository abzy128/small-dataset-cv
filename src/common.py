"""Shared dataset metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLASSES = ["penguin", "hippo", "battery", "controller"]
PROMPTS = {
    "penguin": "a white penguin astronaut figurine",
    "hippo": "a blue hippo toy figurine on a puzzle piece",
    "battery": "an AA battery",
    "controller": "a game controller",
}


def image_meta():
    """Return list of dicts: id, path, label (or 'multi'), domain."""
    rows = []
    for i in range(435, 493):
        if 435 <= i <= 445:
            lab, dom = "penguin", "studio"
        elif 446 <= i <= 457:
            lab, dom = "hippo", "studio"
        elif 458 <= i <= 463:
            lab, dom = "battery", "studio"
        elif 464 <= i <= 475:
            lab, dom = "controller", "studio"
        elif 476 <= i <= 478:
            lab, dom = "controller", "desk"
        elif 479 <= i <= 481:
            lab, dom = "penguin", "desk"
        elif 482 <= i <= 485:
            lab, dom = "hippo", "desk"
        elif 486 <= i <= 488:
            lab, dom = "battery", "desk"
        else:
            lab, dom = "multi", "desk"
        rows.append(dict(id=f"IMG_0{i}", path=str(RAW / f"IMG_0{i}.jpg"), label=lab, domain=dom))
    return rows
