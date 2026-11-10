"""Independent annotation cross-check with Cloudflare Workers AI 'clef' decision models (image modality).
Asks per image: dominant object (choice), per-class presence (noul), scene type (choice). Tracks token usage."""
import base64, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from common import image_meta, ROOT

env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l)
ACC, TOK = env["CF_ACCOUNT_ID"], env["CF_API_TOKEN"]
PRICE = {"clef-flash": 0.038, "clef": 0.24, "clef-omni": 0.15}  # USD per M input tokens
DESC = {
    "penguin": "a white penguin figurine wearing an astronaut-like helmet/hood (red cap) on a white base",
    "hippo": "a small blue hippo toy with a purple crown standing on a teal puzzle-piece base",
    "battery": "an AA battery (copper/black Duracell)",
    "controller": "a mint-green and cream wireless game controller / gamepad",
}
QUESTIONS = {
    "main": {"type": "choice", "instructions": "Which object is the main subject of the photo? Choose 'multiple' if two or more of the listed objects are clearly present.",
             "criteria": {**DESC, "multiple": "two or more of the listed objects appear together"}},
    **{f"has_{c}": {"type": "noul", "instructions": f"Is {d} visible anywhere in the image (even partially)?"} for c, d in DESC.items()},
    "scene": {"type": "choice", "instructions": "What is the background of the photo?",
              "criteria": {"studio": "plain, uniform light background with nothing else", "desk": "cluttered desk with mouse pad, headphones or other items"}},
}

def b64(path, side=896):
    im = Image.open(path).convert("RGB"); im.thumbnail((side, side))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def call(model, path):
    body = json.dumps({"model": model, "state": "A photo for an object-recognition dataset.", "questions": QUESTIONS,
                       "images": [{"content_type": "image/jpeg", "base64": b64(path)}]}).encode()
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{ACC}/ai/run/@cf/cloudflare/{model}", data=body,
                                 headers={"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            t = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=120))
            r["result"]["latency_s"] = time.time() - t
            return r["result"]
        except Exception as e:
            err = e; time.sleep(2 * (attempt + 1))
    return {"error": str(err)}

if __name__ == "__main__":
    models = sys.argv[1].split(",")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    metas = image_meta()[:limit] if limit else image_meta()
    outp = ROOT / "results" / "clef_annotations.json"; outp.parent.mkdir(exist_ok=True)
    out = json.load(open(outp)) if outp.exists() else {}
    for model in models:
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda m: call(model, m["path"]), metas))
        out.setdefault(model, {}).update({m["id"]: r for m, r in zip(metas, res)})
        toks = sum(r.get("usage", {}).get("input_tokens", 0) for r in res)
        errs = sum("error" in r for r in res)
        print(f"{model}: {len(res)} imgs, {errs} errors, {toks} input tokens, ~${toks * PRICE[model] / 1e6:.4f}")
    json.dump(out, open(outp, "w"), indent=1)
