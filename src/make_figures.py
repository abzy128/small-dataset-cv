"""Generate all report figures + summary tables from results/*.json."""
import json, glob, os, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, CLASSES, image_meta

FIG = ROOT / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
# validated reference categorical palette (slots in fixed order) + text/grid tokens
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLS_COL = dict(zip(CLASSES, S[:4]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})
ANN = json.load(open(ROOT / "data" / "annotations.json"))
META = image_meta()

def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight"); plt.close(fig); print("fig", name)

# ---------- 1. dataset overview with GT boxes ----------
def dataset_sheet():
    th, cols = 220, 10; rows = int(np.ceil(len(META) / cols))
    sheet = Image.new("RGB", (cols * th, rows * th), SURF)
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception: font = None
    for k, m in enumerate(META):
        im = Image.open(ROOT / "data" / "img1024" / f"{m['id']}.jpg").convert("RGB").resize((th - 4, th - 4)); d = ImageDraw.Draw(im)
        s = (th - 4) / ANN[m["id"]]["W"]
        for b in ANN[m["id"]]["boxes"]:
            d.rectangle([v * s for v in b["box"]], outline=CLS_COL[b["cls"]], width=3)
        tag = f"{m['id'][4:]} {m['domain']}"
        d.rectangle([0, 0, 92, 19], fill="white"); d.text((3, 2), tag, fill=INK, font=font)
        sheet.paste(im, ((k % cols) * th + 2, (k // cols) * th + 2))
    sheet.save(FIG / "dataset_overview.jpg", quality=88); print("fig dataset_overview.jpg")

def dataset_counts():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    doms = ["studio", "desk"]
    single = {c: [sum(1 for m in META if m["label"] == c and m["domain"] == d) for d in doms] for c in CLASSES}
    x = np.arange(len(CLASSES)); w = 0.36
    for j, (d, col) in enumerate(zip(doms, [S[0], S[1]])):
        v = [single[c][j] for c in CLASSES]
        bars = ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=col, label=f"{d} (single object)")
        for b, val in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, val + 0.2, str(val), ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, CLASSES); ax.set_ylabel("images"); ax.legend(loc="upper left", ncols=2)
    ax.set_title("Single-object images per class and background (+4 multi-object scenes, 16 boxes)", loc="left")
    ax.grid(axis="x", visible=False); save(fig, "dataset_counts.png")

def box_size_hist():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for d, col in [("studio", S[0]), ("desk", S[1])]:
        v = [np.sqrt((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])) / ANN[m["id"]]["W"] * 100
             for m in META if m["domain"] == d for b in ANN[m["id"]]["boxes"]]
        ax.hist(v, bins=np.arange(0, 80, 4), color=col, alpha=0.75, label=f"{d} ({len(v)} boxes)", edgecolor=SURF, linewidth=1.5)
    ax.set_xlabel("object size: sqrt(box area) as % of image side"); ax.set_ylabel("boxes"); ax.legend()
    ax.set_title("Desk objects are ~2.6x smaller than studio objects (median 13.6% vs 34.9% of image side)", loc="left"); save(fig, "box_sizes.png")

# ---------- 2. classification ----------
def cls_figs():
    fr = json.load(open(ROOT / "results" / "cls_frozen.json")); ft = json.load(open(ROOT / "results" / "cls_finetune.json"))
    backbones = ["resnet50_in1k", "efficientnet_b0_in1k", "convnext_tiny_in22k", "dinov2_small", "dinov2_base", "clip_vitb16", "siglip2_base"]
    nice = {"resnet50_in1k": "ResNet-50 (IN-1k)", "efficientnet_b0_in1k": "EfficientNet-B0 (IN-1k)", "convnext_tiny_in22k": "ConvNeXt-T (IN-22k)",
            "dinov2_small": "DINOv2 ViT-S/14", "dinov2_base": "DINOv2 ViT-B/14", "clip_vitb16": "CLIP ViT-B/16", "siglip2_base": "SigLIP2 B/16"}
    # (a) frozen linear-probe: three protocols, grouped horizontal bars
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(backbones)); h = 0.26
    series = [("cv_mean", "5-fold CV, mixed domains", S[0]), ("shift_acc", "studio -> desk, full image", S[1]),
              ("shift_crop_acc", "studio -> desk, object crop", S[2])]
    for j, (k, lab, col) in enumerate(series):
        v = [fr["frozen"][f"{b}|logreg"][k] for b in backbones]
        ax.barh(y + (j - 1) * (h + 0.02), v, h, color=col, label=lab)
        for yy, val in zip(y + (j - 1) * (h + 0.02), v): ax.text(val + 0.01, yy, f"{val:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [nice[b] for b in backbones]); ax.invert_yaxis(); ax.set_xlim(0, 1.12); ax.set_xlabel("accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=8.5)
    ax.set_title("Frozen features + logistic regression", loc="left", pad=28); ax.grid(axis="y", visible=False)
    save(fig, "cls_frozen_protocols.png")
    # (b) all classification methods: CV vs shift dot plot
    rows = []
    for b in backbones:
        for kind in ("logreg", "knn1", "prototype"):
            r = fr["frozen"][f"{b}|{kind}"]; rows.append((f"{nice[b]} · {kind}", r["cv_mean"], r["shift_acc"], "frozen"))
    for kind in ("svm_rbf",):
        r = fr["frozen"][f"colour_hist|{kind}"]; rows.append(("Colour histogram · SVM", r["cv_mean"], r["shift_acc"], "classical"))
    ftn = {"resnet18_ft": "ResNet-18 fine-tuned", "efficientnet_b0_ft": "EfficientNet-B0 fine-tuned", "convnext_tiny_ft": "ConvNeXt-T fine-tuned",
           "smallcnn_scratch": "Small CNN from scratch", "resnet18_scratch": "ResNet-18 from scratch"}
    for k, n in ftn.items():
        if k in ft: rows.append((n, ft[k]["cv_mean"], ft[k]["shift_mean"], "scratch" if "scratch" in k else "finetuned"))
    zs = fr["zero_shot"]
    for k, n in [("clip_vitb16", "CLIP zero-shot"), ("siglip2_base", "SigLIP2 zero-shot")]:
        rows.append((n, zs[k]["all_acc"], zs[k]["desk_acc"], "zeroshot"))
    clef = clef_cls()
    for k, v in clef.items(): rows.append((f"{k} zero-shot (Workers AI)", v["all"], v["desk"], "zeroshot"))
    grp_lab = {"frozen": "frozen", "classical": "classical", "finetuned": "fine-tuned", "scratch": "scratch", "zeroshot": "zero-shot"}
    keep = [r for r in rows if r[3] != "frozen" or r[0].endswith("logreg") or r[0] in ("CLIP ViT-B/16 · knn1", "DINOv2 ViT-S/14 · prototype", "EfficientNet-B0 (IN-1k) · logreg")]
    keep = sorted(keep, key=lambda r: (r[2], r[1]))
    fig, ax = plt.subplots(figsize=(8.2, 0.3 * len(keep) + 1.4)); y = np.arange(len(keep))
    for yy, r in zip(y, keep):
        ax.plot([r[1], r[2]], [yy, yy], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter([r[2] for r in keep], y, s=55, color=S[1], edgecolor=SURF, linewidth=1.5, zorder=3, label="studio -> desk (13 desk images)")
    ax.scatter([r[1] for r in keep], y, s=95, facecolor="none", edgecolor=S[0], linewidth=2.2, zorder=4, label="in-domain (5-fold CV; zero-shot: all single-object images)")
    ax.set_yticks(y, [f"{r[0].replace(' · ', ' / ')}  [{grp_lab[r[3]]}]" for r in keep], fontsize=8.5)
    ax.set_xlim(0.15, 1.03); ax.set_xlabel("accuracy"); ax.axvline(0.25, color=GRID, ls=":", lw=1)
    ax.text(0.255, -0.9, "chance", fontsize=8, color=INK2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncols=1, fontsize=8.5); ax.grid(axis="y", visible=False)
    ax.set_title("Classification: in-domain vs. domain-shift accuracy (sorted by shift accuracy)", loc="left", pad=38)
    save(fig, "cls_cv_vs_shift.png")
    # (c) few-shot curves: small multiples (in-domain / shift)
    pick = ["resnet50_in1k", "convnext_tiny_in22k", "dinov2_small", "clip_vitb16"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in [(axs[0], "fewshot_in", "in-domain (held-out studio, balanced acc.)"), (axs[1], "fewshot_shift", "studio -> desk (13 desk images)")]:
        for b, col in zip(pick, S):
            ks = [1, 2, 3, 5]; m = [fr["frozen"][f"{b}|prototype"][key][str(k)][0] for k in ks]; sd = [fr["frozen"][f"{b}|prototype"][key][str(k)][1] for k in ks]
            ax.plot(ks, m, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=nice[b])
            ax.fill_between(ks, np.array(m) - np.array(sd), np.array(m) + np.array(sd), color=col, alpha=0.08, lw=0)
        ax.set_xticks([1, 2, 3, 5]); ax.set_xlabel("labelled studio images per class (k)"); ax.set_title(title, loc="left"); ax.set_ylim(0.2, 1.02)
    axs[0].set_ylabel("accuracy (mean ± sd, 30 draws)"); axs[0].axhline(0.25, color=GRID, ls=":", lw=1); axs[1].axhline(0.25, color=GRID, ls=":", lw=1)
    axs[1].text(4.9, 0.27, "chance", ha="right", fontsize=8, color=INK2)
    axs[0].legend(loc="lower right", fontsize=8.5); fig.suptitle("Few-shot nearest-prototype classification (frozen features)", x=0.01, y=1.04, ha="left", fontweight="bold")
    save(fig, "cls_fewshot.png")
    # (d) confusion matrices on shift test
    from sklearn.metrics import confusion_matrix
    yt = np.array(fr["y"]); dom = np.array(fr["domain"]); yd = yt[dom == "desk"]
    mats = [("ResNet-18 fine-tuned", ft.get("resnet18_ft", {}).get("shift_pred")), ("CLIP ViT-B/16 kNN (full image)", fr["frozen"]["clip_vitb16|knn1"]["shift_pred"]),
            ("CLIP ViT-B/16 kNN (object crop)", fr["frozen"]["clip_vitb16|knn1"]["shift_crop_pred"])]
    mats = [m for m in mats if m[1] is not None]
    fig, axs = plt.subplots(1, len(mats), figsize=(4.4 * len(mats), 4.0), sharey=True)
    for n_, (ax, (t, p)) in enumerate(zip(np.atleast_1d(axs), mats)):
        cm = confusion_matrix(yd, p, labels=range(4))
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fe", "#2a78d6", "#0d366b"]), vmin=0, vmax=4)
        for i in range(4):
            for j in range(4): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] >= 2 else INK, fontsize=10)
        ax.set_xticks(range(4), CLASSES, rotation=30); ax.set_yticks(range(4), CLASSES); ax.grid(False)
        ax.set_xlabel("predicted"); ax.set_ylabel("true" if n_ == 0 else ""); ax.set_title(f"{t}\nacc {np.mean(np.array(p) == yd):.2f}", fontsize=10)
    fig.suptitle("Confusion on the 13 desk images (trained on studio only)", x=0.01, y=1.06, ha="left", fontweight="bold"); save(fig, "cls_confusion_shift.png")
    # (e) fine-tune training curve (two panels, no dual axis)
    hist = {k: ft[k]["shift_hist"] for k in ("resnet18_ft", "convnext_tiny_ft", "efficientnet_b0_ft") if k in ft and len(ft[k]["shift_hist"]) > 2}
    if hist:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
        for (k, h), col in zip(hist.items(), S):
            e = [r["epoch"] for r in h]
            axs[0].plot(e, [r["train_loss"] for r in h], color=col, lw=2, label=ftn[k]); axs[1].plot(e, [r["test_acc"] for r in h], color=col, lw=2, label=ftn[k])
        axs[0].set_title("training loss (studio)", loc="left"); axs[1].set_title("accuracy on desk test images", loc="left")
        axs[1].axhline(0.25, color=GRID, ls=":"); axs[1].set_ylim(0, 1.02); axs[1].legend(fontsize=8.5)
        for a in axs: a.set_xlabel("epoch")
        save(fig, "cls_finetune_curves.png")
    return rows

def clef_cls():
    d = json.load(open(ROOT / "results" / "clef_annotations.json")); out = {}
    for model, r in d.items():
        single = [m for m in META if m["label"] != "multi"]
        ok = [r[m["id"]]["answers"]["main"]["choice"] == m["label"] for m in single]
        desk = [o for o, m in zip(ok, single) if m["domain"] == "desk"]
        out[model] = dict(all=float(np.mean(ok)), desk=float(np.mean(desk)))
    return out

# ---------- 3. detection ----------
DET_NICE = {"zeroshot_gdino_tiny": "Grounding DINO-T zero-shot", "zeroshot_owlv2": "OWLv2 zero-shot*", "yolo11n_scratch": "YOLO11n from scratch",
            "yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo11n_960": "YOLO11n @960", "yolo11s": "YOLO11s", "frcnn": "Faster R-CNN R50-FPN",
            "yolo11n_cp": "YOLO11n + copy-paste", "yolo11s_cp": "YOLO11s + copy-paste", "frcnn_cp": "Faster R-CNN + copy-paste"}
def det_table():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "det" / "metrics_*.json"))):
        tag = Path(f).stem.replace("metrics_", ""); split, name = tag.split("_", 1); r = json.load(open(f))
        base = name.split("_seed")[0]; seed = int(name.split("_seed")[1]) if "_seed" in name else 0
        rows.append(dict(split=split, name=name, base=base, seed=seed, nice=DET_NICE.get(base, base),
                         **{f"{s}_{k}": r[s][k] for s in ("all", "single", "multi") for k in ("mAP50", "mAP50_95", "mAR100")},
                         pc={c: r["all"]["per_class_AP50_95"].get(c) for c in CLASSES}, train_time=r.get("train_time_s", 0), infer_ms=r.get("infer_ms"), params=r.get("params")))
    opj = json.load(open(ROOT / "results" / "det_operating_point.json"))
    for r in rows:
        o = opj.get(f"{r['split']}_{r['name']}", {}).get("0.5")
        if o: r.update(op_P=o["precision"], op_R=o["recall"], op_F1=o["f1"], op_fp=o["fp"], op_fn=o["fn"])
    return rows

def agg(rows, split, base, key):
    v = [r[key] for r in rows if r["split"] == split and r["base"] == base]
    return (float(np.mean(v)), v) if v else (np.nan, [])

def hbar_with_seeds(ax, rows, split, order, keys, h=0.38):
    y = np.arange(len(order))
    for j, (k, lab, col) in enumerate(keys):
        off = (j - (len(keys) - 1) / 2) * (h + 0.03)
        means = [agg(rows, split, n, k)[0] for n in order]
        ax.barh(y + off, means, h, color=col, label=lab)
        for yy, n, m in zip(y + off, order, means):
            vals = agg(rows, split, n, k)[1]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [yy, yy], color=INK, lw=1.2, zorder=4)
                ax.scatter(vals, [yy] * len(vals), s=12, color=INK, zorder=5)
            if not np.isnan(m):
                ax.text(max(vals) + 0.012, yy, f"{m:.2f}" + (f" (n={len(vals)})" if len(vals) > 1 else ""), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [DET_NICE[n] for n in order]); ax.set_xlim(0, 1.18); ax.grid(axis="y", visible=False)

def det_figs(rows):
    order = [k for k in DET_NICE if any(r["base"] == k for r in rows)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.55 * len(order) + 1.6), sharey=True)
    for ax, split, title in [(axs[0], "S1", "S1: train studio only (41 imgs) -> test 17 imgs"), (axs[1], "S2", "S2: + 8 desk imgs (49) -> test 9 imgs")]:
        hbar_with_seeds(ax, rows, split, order, [("all_mAP50", "mAP@0.5", S[0]), ("all_mAP50_95", "mAP@0.5:0.95", S[1])])
        ax.set_title(title, loc="left")
    axs[0].invert_yaxis(); axs[1].legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, -0.02, "Bars = mean over seeds; black dots/whiskers = individual seeds (where n>1). *OWLv2 seeded part of the labels -> optimistic.", fontsize=8, color=INK2)
    save(fig, "det_map.png")
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(order) + 1.4))
    hbar_with_seeds(ax, rows, "S1", order, [("single_mAP50", "desk, single object (13 imgs)", S[0]), ("multi_mAP50", "multi-object scenes (4 imgs, 16 boxes)", S[2])])
    ax.invert_yaxis(); ax.set_xlabel("mAP@0.5")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=8.5)
    ax.set_title("S1 test subsets", loc="left", pad=24); save(fig, "det_subsets_S1.png")
    M = np.array([[np.mean([r["pc"][c] or 0 for r in rows if r["split"] == "S1" and r["base"] == n]) for c in CLASSES] for n in order])
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(order) + 1.2))
    ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fe", "#2a78d6", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(4): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.5 else INK, fontsize=9)
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(len(order)), [DET_NICE[n] for n in order]); ax.grid(False)
    ax.set_title("S1 per-class AP@0.5:0.95 (mean over seeds)", loc="left"); save(fig, "det_per_class_S1.png")
    # operating point (score >= 0.5)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(order) + 1.4))
    hbar_with_seeds(ax, rows, "S1", order, [("op_P", "precision", S[0]), ("op_R", "recall", S[1]), ("op_F1", "F1", S[2])], h=0.26)
    ax.invert_yaxis(); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=8.5)
    ax.set_title("S1 at a deployment threshold (score >= 0.5, IoU >= 0.5)", loc="left", pad=24); save(fig, "det_operating_point_S1.png")
    # learning curve
    fig, ax = plt.subplots(figsize=(7, 3.6))
    pts = [(4, "yolo11n_k1"), (8, "yolo11n_k2"), (16, "yolo11n_k4"), (41, "yolo11n")]
    for key, lab, col in [("all_mAP50", "mAP@0.5", S[0]), ("all_mAP50_95", "mAP@0.5:0.95", S[1])]:
        xs = [x for x, n in pts]; ms = [agg(rows, "S1", n, key)[0] for x, n in pts]
        ax.plot(xs, ms, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=f"YOLO11n {lab}")
        v = agg(rows, "S1", "yolo11n", key)[1]; ax.scatter([41] * len(v), v, s=14, color=INK, zorder=5)
        cp = agg(rows, "S1", "yolo11n_cp", key)
        ax.scatter([41 * 1.08], [cp[0]], marker="D", s=55, color=col, edgecolor=INK, zorder=6)
    ax.annotate("+ copy-paste (300 synthetic\nimages from the same 41)", (41 * 1.08, agg(rows, "S1", "yolo11n_cp", "all_mAP50")[0]), xytext=(-150, 4),
                textcoords="offset points", fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 41], ["4\n(1/class)", "8\n(2/class)", "16\n(4/class)", "41\n(all studio)"])
    ax.set_ylim(0, 1); ax.set_ylabel("S1 test mAP"); ax.set_xlabel("labelled training images"); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Detection learning curve (studio -> desk)", loc="left"); save(fig, "det_learning_curve.png")
    # cost vs quality
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for n in order:
        v = [r for r in rows if r["split"] == "S1" and r["base"] == n and r["train_time"]]
        if not v: continue
        x = np.mean([r["train_time"] for r in v]) / 60; yv = np.mean([r["all_mAP50_95"] for r in v])
        ax.scatter(x, yv, s=60, color=S[0], edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(DET_NICE[n], (x, yv), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_xlabel("training wall-clock (min, log; M5 Pro GPU, shared between concurrent jobs)"); ax.set_ylabel("S1 mAP@0.5:0.95")
    ax.set_title("Training cost vs. detection quality (S1)", loc="left"); save(fig, "det_cost.png")

def det_examples(models, ids, name, split="S1"):
    fig, axs = plt.subplots(len(models), len(ids), figsize=(3.1 * len(ids), 3.25 * len(models)))
    for r, mname in enumerate(models):
        p = json.load(open(ROOT / "results" / "det" / f"preds_{split}_{mname}.json")) if os.path.exists(ROOT / "results" / "det" / f"preds_{split}_{mname}.json") \
            else json.load(open(ROOT / "results" / "det" / f"preds_{mname}.json"))
        for c, i in enumerate(ids):
            ax = axs[r, c]; ax.imshow(Image.open(ROOT / "data" / "img1024" / f"{i}.jpg")); ax.axis("off")
            for d in p.get(i, []):
                if d["score"] < 0.35: continue
                x0, y0, x1, y1 = d["box"]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=CLS_COL[d["cls"]], lw=2.2))
                ax.text(x0, y0 - 8, f"{d['cls']} {d['score']:.2f}", color="white", fontsize=7, bbox=dict(facecolor=CLS_COL[d["cls"]], pad=1, lw=0))
            if c == 0: ax.text(-40, 512, DET_NICE.get(mname.replace("zeroshot_", "zeroshot_"), mname), rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
            if r == 0: ax.set_title(i, fontsize=9)
    plt.subplots_adjust(wspace=0.03, hspace=0.05); save(fig, name)

def synth_figs():
    # SAM cutouts of studio train objects
    keys = [k for k in sorted(json.load(open(ROOT / "data" / "masks" / "stats.json"))) if int(k.split("_")[1]) <= 475]
    th, cols = 150, 12; rows = int(np.ceil(len(keys) / cols)); sheet = Image.new("RGB", (cols * th, rows * th), (236, 236, 233))
    for n, k in enumerate(keys):
        iid, j = k.rsplit("_", 1); s_ = 1024 / ANN[iid]["W"]; x0, y0, x1, y1 = [int(v * s_) for v in ANN[iid]["boxes"][int(j)]["box"]]
        im = Image.open(ROOT / "data" / "img1024" / f"{iid}.jpg").convert("RGB").crop((x0, y0, x1, y1))
        mk = Image.open(ROOT / "data" / "masks" / f"{k}.png").crop((x0, y0, x1, y1))
        bg = Image.new("RGB", im.size, (236, 236, 233)); bg.paste(im, (0, 0), mk); bg.thumbnail((th - 6, th - 6))
        sheet.paste(bg, ((n % cols) * th + 3, (n // cols) * th + 3))
    sheet.save(FIG / "sam_cutouts.jpg", quality=88); print("fig sam_cutouts.jpg")
    fs = sorted(glob.glob(str(ROOT / "data" / "yolo" / "S1_studio2desk" / "synth" / "images" / "*.jpg")))[:8]
    sheet = Image.new("RGB", (4 * 360, 2 * 360), SURF)
    for n, f in enumerate(fs):
        im = Image.open(f).resize((356, 356)); d = ImageDraw.Draw(im)
        for l in open(f.replace("/images/", "/labels/").replace(".jpg", ".txt")):
            c, x, y, w, h = map(float, l.split())
            d.rectangle([(x - w / 2) * 356, (y - h / 2) * 356, (x + w / 2) * 356, (y + h / 2) * 356], outline=CLS_COL[CLASSES[int(c)]], width=3)
        sheet.paste(im, ((n % 4) * 360 + 2, (n // 4) * 360 + 2))
    sheet.save(FIG / "synth_examples.jpg", quality=88); print("fig synth_examples.jpg")

if __name__ == "__main__":
    synth_figs()
    dataset_sheet(); dataset_counts(); box_size_hist()
    rows = cls_figs()
    json.dump(rows, open(ROOT / "results" / "cls_summary_rows.json", "w"), indent=1)
    drows = det_table(); json.dump(drows, open(ROOT / "results" / "det_summary_rows.json", "w"), indent=1)
    if drows: det_figs(drows)
    ex = ["IMG_0478", "IMG_0481", "IMG_0484", "IMG_0486", "IMG_0489", "IMG_0491"]
    cand = [m for m in ["zeroshot_gdino_tiny", "yolo11n", "yolo11n_cp", "yolo11s", "yolo11s_cp", "frcnn"]
            if (ROOT / "results" / "det" / f"preds_S1_{m}.json").exists() or (ROOT / "results" / "det" / f"preds_{m}.json").exists()]
    det_examples(cand, ex, "det_examples_S1.png")
