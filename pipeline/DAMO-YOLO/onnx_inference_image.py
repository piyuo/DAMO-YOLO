# pipeline/DAMO-YOLO/onnx_inference_image.py

#!/usr/bin/env python3
"""Fixed version of ONNX inference script with improved postprocessing.

This version addresses common issues with ONNX model outputs:
1. Low confidence scores that need proper normalization
2. Poor score calibration after export
3. Too many false positive detections

Usage:
  python pipeline/DAMO-YOLO/onnx_inference_image_fixed.py \
      --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
      --image pipeline/dataset/demo/demo.jpg \
      --conf 0.3 \
      --use-raw-scores  # Try using raw scores instead of softmax
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to import onnxruntime lazily so we can give a helpful message if missing.
try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover - informative error
    raise ImportError("onnxruntime is required. Install with `pip install onnxruntime`.") from e

# Add repo root to path (this file lives in pipeline/DAMO-YOLO)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# We'll import project config lazily inside main so that missing / incompatible
# torch + numpy versions don't crash at import time. This allows a pure-numpy
# fallback path.
HAVE_PARSE_CONFIG = False
def _lazy_parse_config(path):
    global HAVE_PARSE_CONFIG
    if HAVE_PARSE_CONFIG:
        from damo.config.base import parse_config  # type: ignore
        return parse_config(path)
    try:
        from damo.config.base import parse_config  # type: ignore
        HAVE_PARSE_CONFIG = True
        return parse_config(path)
    except Exception as e:
        print(f"[WARN] Could not import full config (torch likely failed). Falling back to built-in defaults. Reason: {e}")
        return None


@dataclass
class PreprocConfig:
    image_max_range: Tuple[int, int]
    flip_prob: float
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]
    keep_ratio: bool


def parse_args():
    ap = argparse.ArgumentParser("Fixed ONNXRuntime person inference")
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'damoyolo_tinynasL25_S.py'),
                    help='Model config (optional). If loading fails, a fallback preprocessing config is used.')
    ap.add_argument('--onnx', default=os.path.join(ROOT, 'pipeline', 'output', 'damoyolo_tinynasL25_S_person.onnx'))
    ap.add_argument('--image', default=os.path.join(ROOT, 'pipeline', 'dataset', 'demo', 'demo.jpg'))
    ap.add_argument('--output', default=os.path.join(ROOT, 'pipeline', 'output'))
    ap.add_argument('--infer-size', type=int, nargs=2, default=[640, 640], help='inference size (h w)')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--nms-iou', type=float, default=0.65, help='NMS IoU threshold')
    ap.add_argument('--use-raw-scores', action='store_true', help='Use raw scores without softmax normalization')
    ap.add_argument('--score-threshold-percentile', type=float, default=95.0,
                    help='Use percentile-based thresholding instead of fixed conf (e.g., 95.0 for top 5%)')
    ap.add_argument('--max-detections', type=int, default=100, help='Maximum number of detections to return')
    ap.add_argument('--legacy-single-class', action='store_true',
                    help='Treat model outputs as (person, background) and filter background.')
    ap.add_argument('--cpu', action='store_true', help='Force CPU execution provider only.')
    ap.add_argument('--no-vis', action='store_true', help='Skip visualization (avoids torch import).')
    ap.add_argument('--pure-np', action='store_true', help='Force pure NumPy preprocessing (ignore config transforms).')
    ap.add_argument('--person-class', type=int, default=0, help='Index of person class (when multi-class).')
    ap.add_argument('--auto-person-class', action='store_true', help='Auto-detect person class by highest mean topK score.')
    ap.add_argument('--no-auto-scale', action='store_true', help='Disable automatic bbox scale detection (always attempt scaling).')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


def build_preproc_cfg(test_cfg) -> PreprocConfig:
    t = test_cfg.augment.transform
    return PreprocConfig(
        image_max_range=t.image_max_range,
        flip_prob=t.flip_prob,
        image_mean=tuple(t.image_mean),
        image_std=tuple(t.image_std),
        keep_ratio=t.keep_ratio,
    )


def default_preproc_cfg() -> PreprocConfig:
    # Fallback values matching typical DAMO-YOLO test config
    return PreprocConfig(
        image_max_range=(640, 640),
        flip_prob=0.0,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        # IMPORTANT: Official test transform uses keep_ratio = False (see damo/config/augmentations.py)
        # Using True introduces letterbox padding and breaks bbox scaling.
        keep_ratio=False,
    )


def preprocess_pure_np(origin_img: np.ndarray, cfg: PreprocConfig, infer_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pure NumPy + PIL preprocessing (no torch)."""
    ih, iw = origin_img.shape[:2]
    th, tw = infer_size
    # Training / official test pipeline uses keep_ratio False by default, i.e. direct warp
    if cfg.keep_ratio:
        # Letterbox path (not recommended for this exported model unless it was trained that way)
        scale = min(th / ih, tw / iw)
        new_h = int(round(ih * scale))
        new_w = int(round(iw * scale))
        img_resized = np.array(Image.fromarray(origin_img).resize((new_w, new_h), Image.BILINEAR))
        canvas = np.zeros((th, tw, 3), dtype=img_resized.dtype)
        canvas[:new_h, :new_w] = img_resized
        arr_img = canvas
    else:
        # Direct resize (stretch) – matches training config keep_ratio=False
        img_resized = np.array(Image.fromarray(origin_img).resize((tw, th), Image.BILINEAR))
        arr_img = img_resized
    # Normalize: to float [0,1] then (x - mean)/std
    arr = arr_img.astype(np.float32) / 255.0
    mean = np.array(cfg.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.image_std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    return arr, (iw, ih)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def improved_postprocess(scores_raw: np.ndarray, bboxes_raw: np.ndarray, num_classes: int,
                        nms_conf: float, nms_iou: float, img_size_wh: Tuple[int, int],
                        legacy_single_class: bool, use_raw_scores: bool = False,
                        score_threshold_percentile: float = None, max_detections: int = 100,
                        debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Improved postprocessing with multiple score normalization strategies."""

    if debug:
        print(f"[DEBUG] Improved postprocess input:")
        print(f"  scores_raw: {scores_raw.shape}, range: [{scores_raw.min():.4f}, {scores_raw.max():.4f}]")
        print(f"  bboxes_raw: {bboxes_raw.shape}, range: [{bboxes_raw.min():.4f}, {bboxes_raw.max():.4f}]")
        print(f"  num_classes: {num_classes}, nms_conf: {nms_conf}, legacy_single_class: {legacy_single_class}")
        print(f"  use_raw_scores: {use_raw_scores}, score_threshold_percentile: {score_threshold_percentile}")

    # Choose score processing strategy
    if use_raw_scores:
        if debug:
            print("[DEBUG] Using raw scores without normalization")
        scores = scores_raw.copy()
    elif scores_raw.max() > 50 or scores_raw.min() < -50:
        if debug:
            print("[DEBUG] Applying sigmoid to scores (detected logits)")
        scores = sigmoid(scores_raw)
    else:
        if debug:
            print("[DEBUG] Applying softmax normalization")
        # Apply softmax for proper probability distribution
        exp_scores = np.exp(scores_raw - scores_raw.max(axis=1, keepdims=True))  # numerical stability
        scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    if debug:
        print(f"[DEBUG] After normalization: scores range [{scores.min():.4f}, {scores.max():.4f}]")

    # Extract person scores
    if legacy_single_class and num_classes == 2:
        if debug:
            print("[DEBUG] Using legacy single class mode (person vs background)")
        person_scores = scores[:, 0]
        labels = np.zeros_like(person_scores, dtype=np.int64)
    else:
        if debug:
            print("[DEBUG] Using multi-class mode (temporary pre-filter, person class chosen later)")
        # Keep per-class for later filtering; return all for now.
        labels = scores.argmax(1)
        # Placeholder; caller will select person class.
        person_scores = scores.max(1)

    # Apply dynamic thresholding if percentile is specified
    if score_threshold_percentile is not None:
        threshold = np.percentile(person_scores, score_threshold_percentile)
        if debug:
            print(f"[DEBUG] Using percentile-based threshold: {threshold:.4f} (p{score_threshold_percentile})")
    else:
        threshold = nms_conf

    if debug:
        print(f"[DEBUG] Person scores: min={person_scores.min():.4f}, max={person_scores.max():.4f}")
        print(f"[DEBUG] Scores >= {threshold:.4f}: {(person_scores >= threshold).sum()} out of {len(person_scores)}")

    # Filter by confidence threshold
    mask = person_scores >= threshold
    if not np.any(mask):
        if debug:
            print("[DEBUG] No detections pass threshold")
        return np.array([]), np.array([]), np.array([])

    bboxes = bboxes_raw[mask]
    scores_f = person_scores[mask]
    labels_f = labels[mask]

    if debug:
        print(f"[DEBUG] After confidence filtering: {len(bboxes)} detections")

    # Sort by confidence and limit detections
    if len(scores_f) > max_detections:
        top_indices = np.argsort(scores_f)[-max_detections:]
        bboxes = bboxes[top_indices]
        scores_f = scores_f[top_indices]
        labels_f = labels_f[top_indices]
        if debug:
            print(f"[DEBUG] Limited to top {max_detections} detections")

    # Apply NMS
    def iou(a, b):
        inter_x1 = np.maximum(a[0], b[0])
        inter_y1 = np.maximum(a[1], b[1])
        inter_x2 = np.minimum(a[2], b[2])
        inter_y2 = np.minimum(a[3], b[3])
        iw = np.maximum(0, inter_x2 - inter_x1)
        ih = np.maximum(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union = area_a + area_b - inter + 1e-9
        return inter / union

    keep_inds = []
    order = scores_f.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        keep_inds.append(i)
        if order.size == 1:
            break
        rem = []
        for j in order[1:]:
            if labels_f[j] != labels_f[i]:
                rem.append(j)
            else:
                if iou(bboxes[i], bboxes[j]) <= nms_iou:
                    rem.append(j)
        order = np.array(rem, dtype=np.int64)

    final_bboxes = bboxes[keep_inds]
    final_scores = scores_f[keep_inds]
    final_labels = labels_f[keep_inds]

    if debug:
        print(f"[DEBUG] After NMS: {len(final_bboxes)} detections")

    return final_bboxes, final_scores, final_labels


def format_lines(bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, legacy_single_class: bool) -> List[str]:
    lines: List[str] = []
    if bboxes.size == 0:
        return lines
    for i, (bb, sc, lb) in enumerate(zip(bboxes, scores, labels)):
        if legacy_single_class and lb != 0:
            continue
        x1, y1, x2, y2 = bb.tolist()
        lines.append(f"person bbox[{i}] (raw_label={lb}): (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}), score={sc:.3f}")
    return lines


def draw_detections(image: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   output_path: str, min_score: float = 0.0) -> str:
    """Draw bounding boxes on image and save to output path."""
    if len(bboxes) == 0:
        # Save original image if no detections
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        return output_path

    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except:
        try:
            # Try system fonts on macOS
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Define colors for drawing
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if score < min_score:
            continue

        x1, y1, x2, y2 = bbox.astype(int)
        color = colors[i % len(colors)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label and confidence
        text = f"person: {score:.3f}"

        # Get text size for background
        try:
            bbox_text = draw.textbbox((x1, y1-25), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Draw background rectangle for text
        draw.rectangle([x1, y1-25, x1+text_width+4, y1], fill=color)

        # Draw text
        draw.text((x1+2, y1-23), text, fill=(255, 255, 255), font=font)

    # Save the image
    pil_image.save(output_path)
    return output_path


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(args.onnx)
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    # Load config unless user forces pure numpy path
    cfg = None if args.pure_np else _lazy_parse_config(args.config)
    if cfg is not None:
        # Force single-class modifications
        try:
            cfg.model.head.num_classes = 1
            if hasattr(cfg.model.head, 'legacy'):
                cfg.model.head.legacy = True
            cfg.dataset.class_names = ['person']
        except Exception:
            pass
        pre_cfg = build_preproc_cfg(cfg.test)
    else:
        pre_cfg = default_preproc_cfg()

    infer_h, infer_w = args.infer_size

    providers = ['CPUExecutionProvider'] if args.cpu else None
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_name = sess.get_inputs()[0].name
    outputs_meta = sess.get_outputs()

    # Debug: Print model I/O info
    if args.debug:
        print(f"[DEBUG] ONNX Model Info:")
        print(f"  Input: {input_name}, shape: {sess.get_inputs()[0].shape}, type: {sess.get_inputs()[0].type}")
        print(f"  Outputs ({len(outputs_meta)}):")
        for i, out in enumerate(outputs_meta):
            print(f"    [{i}] {out.name}: shape {out.shape}, type {out.type}")
        print()

    origin_img = np.asarray(Image.open(args.image).convert('RGB'))
    if cfg is None or args.pure_np:
        img_np, (ow, oh) = preprocess_pure_np(origin_img, pre_cfg, (infer_h, infer_w))
    else:
        # Use project preprocessing if available
        try:
            from damo.structures.image_list import to_image_list  # type: ignore
            from damo.dataset.transforms import transforms as T  # type: ignore
            import torch  # noqa: F401

            transform = [
                T.Resize(pre_cfg.image_max_range, target_size=(infer_h, infer_w), keep_ratio=pre_cfg.keep_ratio),
                T.RandomHorizontalFlip(pre_cfg.flip_prob),
                T.ToTensor(),
                T.Normalize(mean=pre_cfg.image_mean, std=pre_cfg.image_std)
            ]
            transform = T.Compose(transform)
            img, _ = transform(origin_img)
            img_list = to_image_list(img, size_divisibility=0)
            tensors = img_list.tensors  # torch.Tensor shape (1,C,H,W)
            _, _, h, w = tensors.shape
            th, tw = infer_h, infer_w
            if h > th or w > tw:
                raise ValueError(f"Transformed image ({h},{w}) larger than target {(infer_h, infer_w)}")
            if (h, w) != (th, tw):
                pad = torch.zeros((1, tensors.shape[1], th, tw), dtype=tensors.dtype)
                pad[:, :, :h, :w].copy_(tensors)
                tensors = pad
            img_np = tensors.cpu().numpy()
            ow, oh = origin_img.shape[1], origin_img.shape[0]
        except Exception as e:
            if args.debug:
                print(f"[INFO] Falling back to pure NumPy preprocessing: {e}")
            img_np, (ow, oh) = preprocess_pure_np(origin_img, pre_cfg, (infer_h, infer_w))

    ort_out = sess.run(None, {input_name: img_np})
    if len(ort_out) < 2:
        raise RuntimeError(f"Expected at least 2 outputs (scores, boxes); got {len(ort_out)}: {[m.name for m in outputs_meta]}")

    scores_raw = np.asarray(ort_out[0])
    bboxes_raw = np.asarray(ort_out[1])
    if scores_raw.ndim == 3:
        scores_raw = scores_raw[0]
    if bboxes_raw.ndim == 3:
        bboxes_raw = bboxes_raw[0]

    num_classes_guess = scores_raw.shape[1]
    # Do NOT automatically assume legacy just because num_classes==2.
    # Many exported heads have two real classes (legacy=False). Require explicit flag.
    legacy = args.legacy_single_class
    if args.debug and num_classes_guess == 2 and not legacy:
        print("[DEBUG] Detected 2 class scores; treating as multi-class (set --legacy-single-class if second channel is background).")

    bboxes, scores, labels = improved_postprocess(
        scores_raw, bboxes_raw, num_classes_guess, args.conf, args.nms_iou,
        (ow, oh), legacy, args.use_raw_scores, args.score_threshold_percentile,
        args.max_detections, args.debug
    )

    # Decide which class index is 'person' if multi-class and not legacy
    person_class_index = args.person_class
    if not legacy and bboxes.size != 0 and num_classes_guess > 1:
        if args.auto_person_class:
            # Heuristic: treat each class column's top-K mean as signal strength
            K = min(200, scores_raw.shape[0])
            # Softmax baseline if not raw
            probs = scores_raw if args.use_raw_scores else np.exp(scores_raw - scores_raw.max(axis=1, keepdims=True))
            if not args.use_raw_scores:
                probs = probs / probs.sum(axis=1, keepdims=True)
            topk_means = []
            for ci in range(num_classes_guess):
                cls_scores = np.sort(probs[:, ci])[-K:]
                topk_means.append((ci, cls_scores.mean()))
            topk_means.sort(key=lambda x: x[1], reverse=True)
            person_class_index = topk_means[0][0]
            if args.debug:
                print(f"[DEBUG] Auto-selected person class index {person_class_index} via topK mean scores: {topk_means[:3]}")
        if args.debug:
            print(f"[DEBUG] Filtering detections to class index {person_class_index}")
        # Keep only boxes whose predicted label equals chosen class
        keep_mask = labels == person_class_index
        bboxes = bboxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

    # Dynamic scaling detection for bbox coordinates
    if bboxes.size != 0 and not args.no_auto_scale:
        infer_h, infer_w = args.infer_size
        max_before = bboxes[:, [0,2]].max()
        max_after_expected = max_before * (ow / float(infer_w))
        # Heuristic: if more than half of x2 values already exceed infer_w * 1.02, assume boxes already in original scale
        exceed_ratio = (bboxes[:, 2] > (infer_w * 1.02)).mean()
        need_scaling = exceed_ratio < 0.5  # if less than 50% exceed infer_w they are likely still in net scale
        if args.debug:
            print(f"[DEBUG] BBox scaling heuristic: exceed_ratio={exceed_ratio:.2f} need_scaling={need_scaling}")
        if need_scaling:
            scale_x = ow / float(infer_w)
            scale_y = oh / float(infer_h)
            if args.debug:
                print(f"[DEBUG] Rescaling {len(bboxes)} boxes from network size {(infer_w, infer_h)} to original {(ow, oh)} with scales (x={scale_x:.4f}, y={scale_y:.4f})")
            bboxes[:, [0, 2]] *= scale_x
            bboxes[:, [1, 3]] *= scale_y
        else:
            if args.debug:
                print("[DEBUG] Skipping bbox scaling; boxes appear already in original image space.")

    # Clamp boxes to image bounds
    if bboxes.size != 0:
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, ow - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, oh - 1)

    # If user mistakenly requested raw scores and they are tiny, hint about softmax.
    if args.use_raw_scores and scores.size != 0 and scores.max() < 0.2 and not args.debug:
        print("[HINT] Raw logits are very small. Try removing --use-raw-scores to apply softmax normalization.")

    lines = format_lines(bboxes, scores, labels, legacy)
    if not lines:
        print(f"No person detections found with current settings")
        print(f"Try using --use-raw-scores or --score-threshold-percentile 90")
    else:
        print(f"Found {len(lines)} person detections:")
        for l in lines:
            print(l)

    # Save visualization image
    if not args.no_vis:
        input_filename = os.path.basename(args.image)
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_result{ext}"
        output_path = os.path.join(args.output, output_filename)

        try:
            saved_path = draw_detections(origin_img, bboxes, scores, labels, output_path)
            print(f"📸 Saved visualization to: {saved_path}")
        except Exception as e:
            print(f"❌ Failed to save visualization: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # Also try the original visualization if config is available
    if not args.no_vis and cfg is not None:
        try:
            from tools.demo import Infer  # type: ignore
            infer_vis = Infer(cfg, infer_size=[infer_h, infer_w], device='cpu', ckpt=args.onnx, output_dir=args.output)
            infer_vis.engine_type = 'onnx'
            infer_vis.visualize(origin_img, bboxes, scores, labels, conf=args.conf,
                               save_name=os.path.basename(args.image), save_result=True)
            print(f"📸 Also saved original visualization method result")
        except Exception as e:
            if args.debug:
                print(f"[DEBUG] Original visualization failed: {e}")
            # Don't show error to user since we already have our simple visualization
if __name__ == '__main__':
    main()