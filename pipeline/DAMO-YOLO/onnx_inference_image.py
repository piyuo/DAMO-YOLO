#!/usr/bin/env python3
"""Run ONNXRuntime inference on a single image with a person-only DAMO-YOLO model.

Model: pipeline/output/damoyolo_tinynasL25_S_person.onnx
Input image: pipeline/dataset/demo/demo.jpg

The exported ONNX is assumed to be the same checkpoint as used in
`py_inference_image.py` (single person class, legacy layout with background channel).

This script purposely avoids importing heavy torch runtime pieces; only numpy +
onnxruntime + Pillow and minimal project utilities are used. It replicates the
essential preprocessing steps from `tools/demo.py` -> `transform_img`.

Usage (from repo root):
  python pipeline/DAMO-YOLO/onnx_inference_image.py \
	  --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
	  --image pipeline/dataset/demo/demo.jpg \
	  --output pipeline/output \
	  --conf 0.3

Output: annotated image saved beside output dir + printed detections.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

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
	ap = argparse.ArgumentParser("ONNXRuntime person inference")
	ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'damoyolo_tinynasL25_S.py'),
					help='Model config (optional). If loading fails, a fallback preprocessing config is used.')
	ap.add_argument('--onnx', default=os.path.join(ROOT, 'pipeline', 'output', 'damoyolo_tinynasL25_S_person.onnx'))
	ap.add_argument('--image', default=os.path.join(ROOT, 'pipeline', 'dataset', 'demo', 'demo.jpg'))
	ap.add_argument('--output', default=os.path.join(ROOT, 'pipeline', 'output'))
	ap.add_argument('--infer-size', type=int, nargs=2, default=[640, 640], help='inference size (h w)')
	ap.add_argument('--conf', type=float, default=0.3)
	ap.add_argument('--legacy-single-class', action='store_true',
					help='Treat model outputs as (person, background) and filter background.')
	ap.add_argument('--cpu', action='store_true', help='Force CPU execution provider only.')
	ap.add_argument('--no-vis', action='store_true', help='Skip visualization (avoids torch import).')
	ap.add_argument('--pure-np', action='store_true', help='Force pure NumPy preprocessing (ignore config transforms).')
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
		keep_ratio=True,
	)


def preprocess_pure_np(origin_img: np.ndarray, cfg: PreprocConfig, infer_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
	"""Pure NumPy + PIL preprocessing (no torch)."""
	ih, iw = origin_img.shape[:2]
	th, tw = infer_size
	if cfg.keep_ratio:
		scale = min(th / ih, tw / iw)
		new_h = int(round(ih * scale))
		new_w = int(round(iw * scale))
	else:
		new_h, new_w = th, tw
	img_resized = np.array(Image.fromarray(origin_img).resize((new_w, new_h), Image.BILINEAR))
	canvas = np.zeros((th, tw, 3), dtype=img_resized.dtype)
	canvas[:new_h, :new_w] = img_resized
	# Normalize: to float [0,1] then (x - mean)/std
	arr = canvas.astype(np.float32) / 255.0
	mean = np.array(cfg.image_mean, dtype=np.float32).reshape(1, 1, 3)
	std = np.array(cfg.image_std, dtype=np.float32).reshape(1, 1, 3)
	arr = (arr - mean) / std
	# HWC -> CHW
	arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
	return arr, (iw, ih)


def preprocess_with_config(origin_img: np.ndarray, cfg: PreprocConfig, infer_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
	"""Try to use project's transform pipeline (will import torch)."""
	try:
		from damo.structures.image_list import to_image_list  # type: ignore
		from damo.dataset.transforms import transforms as T  # type: ignore
		import torch  # noqa: F401
	except Exception as e:
		print(f"[INFO] Falling back to pure NumPy preprocessing: {e}")
		return preprocess_pure_np(origin_img, cfg, infer_size)

	transform = [
		T.Resize(cfg.image_max_range, target_size=infer_size, keep_ratio=cfg.keep_ratio),
		T.RandomHorizontalFlip(cfg.flip_prob),
		T.ToTensor(),
		T.Normalize(mean=cfg.image_mean, std=cfg.image_std)
	]
	transform = T.Compose(transform)
	img, _ = transform(origin_img)
	img_list = to_image_list(img, size_divisibility=0)
	tensors = img_list.tensors  # torch.Tensor shape (1,C,H,W)
	_, _, h, w = tensors.shape
	th, tw = infer_size
	if h > th or w > tw:
		raise ValueError(f"Transformed image ({h},{w}) larger than target {infer_size}")
	if (h, w) != (th, tw):
		import torch
		pad = torch.zeros((1, tensors.shape[1], th, tw), dtype=tensors.dtype)
		pad[:, :, :h, :w].copy_(tensors)
		tensors = pad
	arr = tensors.cpu().numpy()
	return arr, (origin_img.shape[1], origin_img.shape[0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess(scores_raw: np.ndarray, bboxes_raw: np.ndarray, num_classes: int, nms_conf: float, nms_iou: float,
				img_size_wh: Tuple[int, int], legacy_single_class: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Mimic damo.utils.postprocess for single-image batch.

	scores_raw: (N, num_classes) before softmax? In exported ONNX these may be logits.
	bboxes_raw: (N, 4) in xyxy format (already scaled to model input pad region).
	Returns filtered (bboxes, scores, labels).
	"""
	# Apply sigmoid if values look like logits (heuristic).
	if scores_raw.max() > 50 or scores_raw.min() < -50:
		scores = sigmoid(scores_raw)
	else:
		scores = scores_raw

	if legacy_single_class and num_classes == 2:
		person_scores = scores[:, 0]
		labels = np.zeros_like(person_scores, dtype=np.int64)
		mask = person_scores >= nms_conf
		bboxes = bboxes_raw[mask]
		scores_f = person_scores[mask]
		labels = labels[mask]
	else:
		label_inds = scores.argmax(1)
		confs = scores.max(1)
		mask = confs >= nms_conf
		bboxes = bboxes_raw[mask]
		scores_f = confs[mask]
		labels = label_inds[mask]

	if bboxes.shape[0] == 0:
		return bboxes, scores_f, labels

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
			if labels[j] != labels[i]:
				rem.append(j)
			else:
				if iou(bboxes[i], bboxes[j]) <= nms_iou:
					rem.append(j)
		order = np.array(rem, dtype=np.int64)
	return bboxes[keep_inds], scores_f[keep_inds], labels[keep_inds]


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

	origin_img = np.asarray(Image.open(args.image).convert('RGB'))
	if cfg is None or args.pure_np:
		img_np, (ow, oh) = preprocess_pure_np(origin_img, pre_cfg, (infer_h, infer_w))
	else:
		img_np, (ow, oh) = preprocess_with_config(origin_img, pre_cfg, (infer_h, infer_w))

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
	legacy = args.legacy_single_class or (num_classes_guess == 2)
	bboxes, scores, labels = postprocess(scores_raw, bboxes_raw, num_classes_guess, args.conf, 0.65,
										 (ow, oh), legacy)

	lines = format_lines(bboxes, scores, labels, legacy)
	if not lines:
		print(f"No person detections >= conf {args.conf}")
	else:
		print("Detections (filtered):")
		for l in lines:
			print(l)

	if args.no_vis:
		return
	if cfg is None:
		if args.debug:
			print("[DEBUG] Skipping visualization (config unavailable without torch). Use --no-vis to silence.")
		return
	try:
		from tools.demo import Infer  # type: ignore
		infer_vis = Infer(cfg, infer_size=[infer_h, infer_w], device='cpu', ckpt=args.onnx, output_dir=args.output)
		infer_vis.engine_type = 'onnx'
		infer_vis.visualize(origin_img, bboxes, scores, labels, conf=args.conf,
							 save_name=os.path.basename(args.image), save_result=True)
	except Exception as e:
		if args.debug:
			print(f"[DEBUG] Visualization failed: {e}")


if __name__ == '__main__':
	main()
