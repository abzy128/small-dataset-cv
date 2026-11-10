<img src="assets/icon.svg" width="64" height="64" alt="">

# small-dataset-ml

How far can you get with **58 phone photos**? This repo labels a tiny dataset and trains classifiers and object detectors on it, all locally on an Apple M5 Pro. It then asks which tricks actually help when data is scarce.

Full write-up with every number, chart and caveat: **[report/REPORT.md](report/REPORT.md)**.

![dataset](report/figures/dataset_overview.jpg)

**The data:** 4 objects (penguin figurine, hippo toy, AA battery, game controller). There are 41 close-ups on a plain **studio** background, 13 shots on a cluttered **desk**, and 4 scenes with all four objects. Desk objects are about 2.6× smaller in the frame.

**The labels:** 70 bounding boxes from OWLv2 proposals, a colour-mask rule and 5 manual fixes, all reviewed visually. Cloudflare Workers AI's `clef` decision model independently agreed with 58/58 image labels, for about $0.04 of API spend in total.

## Key findings

**1. Random cross-validation lies.** Nearly every classifier scores 96–100% in shuffled 5-fold CV. Trained on studio photos and tested on desk photos, the same models fall to 23–85%. Given a crop around the object, *every* backbone gets 13/13. The models recognise the objects fine; they fail to *find* them.

![classification](report/figures/cls_frozen_protocols.png)

**2. Fine-tuned CNNs learn shortcuts.** A fine-tuned ResNet-18 labels every desk photo "controller". Grad-CAM shows why: it looks at a mint-green USB stick that matches the controller's colour, not at the object.

![gradcam](report/figures/gradcam_shortcut.png)

**3. Frozen foundation-model features beat fine-tuning at this scale.** CLIP, DINOv2 and SigLIP2 embeddings with kNN or class prototypes reach 90–98% from **one** image per class and hold up best under the background shift. Models trained from scratch are at chance on the desk.

**4. For detection, copy-paste augmentation was the most effective change.** Cutting the training objects out with SAM and pasting them at random sizes onto the same training photos (no new labels) moved YOLO11n from mAP@0.5 0.58 ± 0.11 to **0.85 ± 0.01**. F1 at a 0.5 confidence threshold went from 0.46 to 0.83.

![synthetic](report/figures/synth_examples.jpg)
![learning curve](report/figures/det_learning_curve.png)

**5. High mAP ≠ deployable.** Faster R-CNN has the best studio→desk mAP (0.91), but at a 0.5 confidence threshold it hallucinates objects in the clutter (precision 0.43). YOLO11n with copy-paste gives F1 0.83 at about 10 ms per image, against 107 ms for Faster R-CNN.

![detections](report/figures/det_examples_S1.png)

**6. A few target-domain labels beat everything.** Adding just 8 desk photos (2 per class) brings detectors to 0.95–1.00 mAP@0.5.

| | studio → desk result |
|---|---|
| Best classifier, full image | 0.85 accuracy (CLIP kNN, DINOv2 prototype, EfficientNet-B0 frozen) |
| Classifier on object crop | **1.00** (all backbones) |
| Best deployable detector | YOLO11n + copy-paste: mAP@0.5 0.85, F1 0.83, ~10 ms per image |
| Zero-shot detector (Grounding DINO-T) | mAP@0.5 0.48 |

The desk test set is small (13 images, roughly ±25 points 95% CI), so only the large, consistent effects above are worth trusting. See [Limitations](report/REPORT.md#7-limitations-and-threats-to-validity).

## Layout

```
src/        annotation, feature extraction, training, evaluation, figures
data/       annotations.json, SAM masks, 1024px images (raw HEIC/JPEG and YOLO sets are git-ignored)
results/    metrics, raw predictions, clef responses, logs
report/     REPORT.md + figures/
```

## Reproduce

```bash
uv sync
uv run python src/compose_annotations.py          # labels (after src/propose_boxes*.py)
uv run python src/classify_frozen.py              # frozen-feature classification (after src/features.py)
uv run python src/finetune_cls.py                 # fine-tuned / from-scratch classifiers
uv run python src/build_yolo.py && uv run python src/copy_paste.py
./src/run_det_queue.sh results/det_queue1.txt     # YOLO runs; Faster R-CNN: src/train_frcnn.py
uv run python src/make_figures.py
```

`src/clef_annotate.py` needs a git-ignored `.env` with `CF_ACCOUNT_ID` and `CF_API_TOKEN`. The full step list is in the [report](report/REPORT.md#8-reproducing).
