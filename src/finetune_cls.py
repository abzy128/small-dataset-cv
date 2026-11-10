"""End-to-end fine-tuning (ImageNet-pretrained) vs training from scratch, for 4-class image classification.
Protocols: stratified 5-fold CV (2 repeats) over 54 single-object images, and studio->desk shift. Strong augmentation."""
import json, sys, time, numpy as np, torch, torch.nn as nn, timm
from PIL import Image
import torchvision.transforms.v2 as T
from sklearn.model_selection import RepeatedStratifiedKFold
from common import image_meta, CLASSES, ROOT

dev = "mps"; torch.manual_seed(0)
metas = [m for m in image_meta() if m["label"] != "multi"]
y = np.array([CLASSES.index(m["label"]) for m in metas]); dom = np.array([m["domain"] for m in metas])
imgs = [Image.open(ROOT / "data" / "img1024" / (m["id"] + ".jpg")).convert("RGB").resize((320, 320)) for m in metas]
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([T.ToImage(), T.RandomResizedCrop(224, scale=(0.15, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
                      T.RandomRotation(20), T.ColorJitter(0.3, 0.3, 0.2, 0.03), T.ToDtype(torch.float32, scale=True), norm])
test_tf = T.Compose([T.ToImage(), T.Resize(224), T.ToDtype(torch.float32, scale=True), norm])

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
        self.h = nn.Linear(256, 4)
    def forward(self, x): return self.h(self.f(x))

MODELS = {
    "resnet18_ft": lambda: timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=4),
    "efficientnet_b0_ft": lambda: timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=4),
    "convnext_tiny_ft": lambda: timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=4),
    "smallcnn_scratch": SmallCNN,
    "resnet18_scratch": lambda: timm.create_model("resnet18", pretrained=False, num_classes=4),
}
EPOCHS = {"smallcnn_scratch": 120, "resnet18_scratch": 120}

def train_eval(name, tr, te, log=False):
    m = MODELS[name]().to(dev)
    ep = EPOCHS.get(name, 40); lr = 1e-3 if "scratch" in name else 3e-4
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep * int(np.ceil(len(tr) / 16)))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); hist = []
    Xte = torch.stack([test_tf(imgs[i]) for i in te]).to(dev)
    for e in range(ep):
        m.train(); perm = np.random.permutation(tr); tl = 0
        for b in range(0, len(perm), 16):
            ids = perm[b:b + 16]
            x = torch.stack([train_tf(imgs[i]) for i in ids]).to(dev); t = torch.tensor(y[ids]).to(dev)
            loss = lossf(m(x), t); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(ids)
        if log or e == ep - 1:
            m.eval()
            with torch.no_grad():
                p = m(Xte).argmax(1).cpu().numpy()
            hist.append(dict(epoch=e + 1, train_loss=tl / len(tr), test_acc=float((p == y[te]).mean())))
    return p, hist

def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    outp = ROOT / "results" / "cls_finetune.json"
    res = json.load(open(outp)) if outp.exists() else {}
    for name in names:
        t0 = time.time(); r = {}
        accs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(y, y):
            p, _ = train_eval(name, tr, te); accs.append(float((p == y[te]).mean()))
        r["cv_mean"], r["cv_std"], r["cv_folds"] = float(np.mean(accs)), float(np.std(accs)), accs
        tr, te = np.where(dom == "studio")[0], np.where(dom == "desk")[0]
        shift = []
        for seed in range(3):
            torch.manual_seed(seed); np.random.seed(seed)
            p, hist = train_eval(name, tr, te, log=(seed == 0)); shift.append(float((p == y[te]).mean()))
            if seed == 0: r["shift_hist"], r["shift_pred"] = hist, p.tolist()
        r["shift_accs"] = shift; r["shift_mean"] = float(np.mean(shift)); r["time_s"] = time.time() - t0
        res[name] = r; json.dump(res, open(outp, "w"), indent=1)
        print(f"{name:20s} cv={r['cv_mean']:.3f}±{r['cv_std']:.3f} shift={r['shift_mean']:.3f} {shift} ({r['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
