"""Classification with frozen features (linear probe / kNN / prototypes), colour-histogram SVM, and zero-shot CLIP/SigLIP.
Protocols:
  cv      : repeated stratified 5-fold CV over all 54 single-object images (10 repeats)
  shift   : train on 41 studio images, test on 13 desk images
  fewshot_shift : k labelled studio images per class (k=1,2,3,5), test on the fixed 13 desk images; 30 random draws
  fewshot_in    : same k-shot draws, tested on the unused studio images (balanced accuracy, since class counts vary)
  shift_crop    : like shift, but both train and test use object-centred crops around the GT box (oracle localisation)
"""
import json, numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from common import image_meta, CLASSES, ROOT

metas = [m for m in image_meta() if m["label"] != "multi"]
idx_all = {m["id"]: i for i, m in enumerate(image_meta())}
sel = np.array([idx_all[m["id"]] for m in metas])
y = np.array([CLASSES.index(m["label"]) for m in metas])
dom = np.array([m["domain"] for m in metas])
FEATS = ["resnet50_in1k", "efficientnet_b0_in1k", "convnext_tiny_in22k", "clip_vitb16", "siglip2_base", "dinov2_small", "dinov2_base"]

from sklearn.metrics import balanced_accuracy_score

def boot_ci(correct, n=2000, seed=0):
    r = np.random.default_rng(seed); c = np.asarray(correct, float)
    b = [c[r.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def load(name, crop=False):
    f = np.load(ROOT / "results" / "features" / f"{name}{'_crop' if crop else ''}.npy")[sel]  # N x 2 x D (orig, hflip)
    return normalize(f.reshape(-1, f.shape[-1])).reshape(f.shape)

def colour_hist():
    out = []
    for m in metas:
        hsv = np.asarray(Image.open(ROOT / "data" / "img1024" / (m["id"] + ".jpg")).convert("HSV").resize((256, 256)))
        h, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(12, 4, 4), range=((0, 256),) * 3)
        h = h.ravel() / h.sum(); out.append(np.sqrt(h))
    f = np.array(out); return np.stack([f, f], 1)

def clf_factory(kind):
    if kind == "logreg":
        return lambda: LogisticRegression(C=10, max_iter=5000)
    if kind == "svm_rbf":
        return lambda: make_pipeline(StandardScaler(), SVC(C=10))
    raise ValueError

def fit_predict(kind, F, tr, te):
    if kind in ("logreg", "svm_rbf"):
        X = F[tr].reshape(-1, F.shape[-1]); Y = np.repeat(y[tr], 2)  # hflip as augmentation
        c = clf_factory(kind)(); c.fit(X, Y); return c.predict(F[te, 0])
    if kind == "knn1":
        s = F[te, 0] @ F[tr].reshape(-1, F.shape[-1]).T
        return np.repeat(y[tr], 2)[s.argmax(1)]
    if kind == "prototype":
        P = np.stack([F[tr][y[tr] == c].mean((0, 1)) for c in range(4)])
        return (F[te, 0] @ P.T).argmax(1)

def run(kind, F, rng_seed=0, Fc=None):
    res = {}
    # CV
    accs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=rng_seed).split(y, y):
        accs.append((fit_predict(kind, F, tr, te) == y[te]).mean())
    res["cv_mean"], res["cv_std"] = float(np.mean(accs)), float(np.std(accs))
    # shift
    tr, te = np.where(dom == "studio")[0], np.where(dom == "desk")[0]
    p = fit_predict(kind, F, tr, te)
    res["shift_acc"] = float((p == y[te]).mean()); res["shift_pred"] = p.tolist(); res["shift_ci"] = boot_ci(p == y[te])
    if Fc is not None:
        pc = fit_predict(kind, Fc, tr, te)
        res["shift_crop_acc"] = float((pc == y[te]).mean()); res["shift_crop_pred"] = pc.tolist()
    # few-shot
    rng = np.random.default_rng(rng_seed); studio = np.where(dom == "studio")[0]; desk = np.where(dom == "desk")[0]
    res["fewshot_shift"], res["fewshot_in"] = {}, {}
    for k in (1, 2, 3, 5):
        a, b = [], []
        for _ in range(30):
            tr = np.concatenate([rng.choice(studio[y[studio] == c], k, replace=False) for c in range(4)])
            kk = kind if kind != "svm_rbf" or k > 1 else "knn1"
            a.append((fit_predict(kk, F, tr, desk) == y[desk]).mean())
            te = np.setdiff1d(studio, tr)
            b.append(balanced_accuracy_score(y[te], fit_predict(kk, F, tr, te)))
        res["fewshot_shift"][k] = [float(np.mean(a)), float(np.std(a))]
        res["fewshot_in"][k] = [float(np.mean(b)), float(np.std(b))]
    return res

def zero_shot():
    """CLIP / SigLIP2 zero-shot with text prompts (no training)."""
    from transformers import AutoModel, AutoProcessor
    prompts = {"penguin": "a photo of a white penguin figurine", "hippo": "a photo of a blue hippo toy",
               "battery": "a photo of an AA battery", "controller": "a photo of a game controller"}
    out = {}
    for key, name in [("clip_vitb16", "openai/clip-vit-base-patch16"), ("siglip2_base", "google/siglip2-base-patch16-224")]:
        m = AutoModel.from_pretrained(name).eval(); p = AutoProcessor.from_pretrained(name)
        kw = {"padding": "max_length", "max_length": 64} if "siglip" in name else {"padding": True}
        with torch.no_grad():
            t = m.get_text_features(**p(text=[prompts[c] for c in CLASSES], return_tensors="pt", **kw))
            t = t.pooler_output if hasattr(t, "pooler_output") else t
        T = normalize(t.numpy()); F = load(key)
        pred = (F[:, 0] @ T.T).argmax(1)
        out[key] = dict(all_acc=float((pred == y).mean()), studio_acc=float((pred == y)[dom == "studio"].mean()),
                        desk_acc=float((pred == y)[dom == "desk"].mean()), pred=pred.tolist())
    return out

results = {}
for fname in FEATS:
    F = load(fname); Fc = load(fname, crop=True)
    for kind in ("logreg", "knn1", "prototype"):
        r = run(kind, F, Fc=Fc); results[f"{fname}|{kind}"] = r
        print(f"{fname:22s} {kind:10s} cv={r['cv_mean']:.3f}±{r['cv_std']:.3f} shift={r['shift_acc']:.3f} crop={r['shift_crop_acc']:.3f} "
              f"fsS1={r['fewshot_shift'][1][0]:.3f} fsS5={r['fewshot_shift'][5][0]:.3f} fsI1={r['fewshot_in'][1][0]:.3f} fsI5={r['fewshot_in'][5][0]:.3f}", flush=True)
F = colour_hist()
for kind in ("svm_rbf", "knn1"):
    r = run(kind, F); results[f"colour_hist|{kind}"] = r
    print(f"{'colour_hist':22s} {kind:10s} cv={r['cv_mean']:.3f}±{r['cv_std']:.3f} shift={r['shift_acc']:.3f} "
          f"fsS1={r['fewshot_shift'][1][0]:.3f} fsS5={r['fewshot_shift'][5][0]:.3f} fsI1={r['fewshot_in'][1][0]:.3f} fsI5={r['fewshot_in'][5][0]:.3f}")
zs = zero_shot(); print("zero-shot", {k: (v["all_acc"], v["desk_acc"]) for k, v in zs.items()})
json.dump(dict(frozen=results, zero_shot=zs, ids=[m["id"] for m in metas], y=y.tolist(), domain=dom.tolist()),
          open(ROOT / "results" / "cls_frozen.json", "w"), indent=1)
