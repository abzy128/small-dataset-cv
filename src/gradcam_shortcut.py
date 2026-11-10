"""Shortcut-learning analysis: train ResNet-18 on studio images (same recipe as finetune_cls), then Grad-CAM on desk images.
Also measures the fraction of dark pixels per class in studio images (a spurious cue that correlates with 'controller')."""
import json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import finetune_cls as fc  # reuses data/transforms/model definitions (module-level only; main loop guarded below)
from common import CLASSES, ROOT

torch.manual_seed(0); np.random.seed(0)
y, dom, imgs = fc.y, fc.dom, fc.imgs
tr, te = np.where(dom == "studio")[0], np.where(dom == "desk")[0]
m = fc.MODELS["resnet18_ft"]().to(fc.dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05); lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=40 * 3)
for e in range(40):
    m.train(); perm = np.random.permutation(tr)
    for b in range(0, len(perm), 16):
        ids = perm[b:b + 16]; x = torch.stack([fc.train_tf(imgs[i]) for i in ids]).to(fc.dev)
        loss = lossf(m(x), torch.tensor(y[ids]).to(fc.dev)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
m.eval()
acts, grads = {}, {}
m.layer4.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
m.layer4.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("g", go[0]))
def cam(i):
    x = fc.test_tf(imgs[i])[None].to(fc.dev).requires_grad_(True)
    out = m(x); c = int(out.argmax()); m.zero_grad(); out[0, c].backward()
    w = grads["g"].mean((2, 3), keepdim=True); h = F.relu((w * acts["a"]).sum(1))[0]
    h = F.interpolate(h[None, None], size=(320, 320), mode="bilinear")[0, 0].detach().cpu().numpy()
    return c, torch.softmax(out, 1)[0, c].item(), h / (h.max() + 1e-8)
show = list(te[[0, 3, 7, 10]]) + [i for i in tr if y[i] == 3][2:4] + [i for i in tr if y[i] == 0][:1] + [i for i in tr if y[i] == 1][:1]
fig, axs = plt.subplots(2, 4, figsize=(14, 8.2))
for ax, i in zip(axs.ravel(), show):
    c, p, h = cam(i)
    ax.imshow(imgs[i]); ax.imshow(h, cmap="jet", alpha=0.42); ax.axis("off")
    ax.set_title(f"{fc.metas[i]['id']} ({dom[i]})\ntrue={CLASSES[y[i]]}  pred={CLASSES[c]} ({p:.2f})", fontsize=10,
                 color="#b00020" if c != y[i] else "#1b5e20")
fig.suptitle("Grad-CAM, ResNet-18 fine-tuned on studio images only (top: desk test images, bottom: studio train images)", fontsize=12)
plt.tight_layout(h_pad=2.2); out = ROOT / "report" / "figures"; out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "gradcam_shortcut.png", dpi=110); print("saved")
# dark-pixel statistic
dark = {}
for c in range(4):
    v = [(np.asarray(imgs[i].convert("HSV"))[..., 2] < 60).mean() for i in tr if y[i] == c]
    dark[CLASSES[c]] = dict(mean=float(np.mean(v)), frac_imgs_gt5pct=float(np.mean(np.array(v) > 0.05)))
dark["desk_test"] = dict(mean=float(np.mean([(np.asarray(imgs[i].convert("HSV"))[..., 2] < 60).mean() for i in te])))
json.dump(dark, open(ROOT / "results" / "dark_pixel_stats.json", "w"), indent=1); print(dark)
