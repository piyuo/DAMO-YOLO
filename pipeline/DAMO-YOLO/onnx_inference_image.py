#!/usr/bin/env python3
"""ONNX person-only inference helper script.

This mirrors the functionality of `py_inference_image.py` but loads an ONNX
engine instead of a PyTorch checkpoint and adds flexible score thresholding
options that work well when raw scores are very low or highly skewed.

Example (as used in the accompanying shell script):
	python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
		--onnx pipeline/DAMO-YOLO/input/damoyolo_tinynasL25_S_person.onnx \
		--image pipeline/dataset/demo/demo.jpg \
		--output pipeline/output \
		--use-raw-scores \
		--score-threshold-percentile 99.5 \
		--max-detections 12 \
		--pure-np

Outputs an annotated image saved into the output directory and prints the
filtered detections (person class only). Background predictions are filtered
out for legacy single-class exports where num_classes=1 but the head exports
an extra background channel (legacy=True).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# Script location: repo_root/pipeline/DAMO-YOLO -> go two levels up to reach root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from damo.config.base import parse_config  # noqa: E402
from tools.demo import Infer  # noqa: E402
from damo.utils.demo_utils import transform_img  # noqa: E402
from damo.structures.image_list import ImageList  # noqa: E402
from damo.utils.boxes import postprocess  # noqa: E402


def build_person_config(config_path: str):
	"""Load base config and adapt for single-class (person) legacy inference.

	Adjustments:
	  - head.num_classes = 1
	  - head.legacy = True (if attribute exists) so output has (num_classes + 1) channels.
	  - dataset.class_names = ['person']
	  - enforce test transform size 640x640
	"""
	cfg = parse_config(config_path)
	cfg.model.head.num_classes = 1
	try:
		cfg.model.head.legacy = True
	except Exception:
		pass
	cfg.dataset.class_names = ['person']
	if hasattr(cfg.test, 'augment') and hasattr(cfg.test.augment, 'transform'):
		cfg.test.augment.transform.image_max_range = (640, 640)
	# Lower default NMS confidence threshold so we can apply custom filtering later
	if hasattr(cfg.model.head, 'nms_conf_thre'):
		cfg.model.head.nms_conf_thre = 0.01
	return cfg


def parse_args():
	parser = argparse.ArgumentParser("ONNX person inference")
	parser.add_argument('--config', default=os.path.join(ROOT, 'configs', 'damoyolo_tinynasL25_S.py'),
						help='Base config path (will be adapted to 1-class).')
	parser.add_argument('--onnx', required=True,
						help='Path to ONNX model file (.onnx)')
	parser.add_argument('--image', default=os.path.join(ROOT, 'pipeline', 'dataset', 'demo', 'demo.jpg'),
						help='Input image path')
	parser.add_argument('--output', default=os.path.join(ROOT, 'pipeline', 'output'),
						help='Directory to save annotated image')
	parser.add_argument('--infer-size', type=int, nargs=2, default=[640, 640],
						help='Inference size (h w) fed to the model')
	parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
						help='Device preference (auto picks cuda if available)')
	parser.add_argument('--score-threshold', type=float, default=None,
						help='Explicit score threshold (overrides percentile if set)')
	parser.add_argument('--score-threshold-percentile', type=float, default=None,
						help='Percentile of score distribution to use as threshold (e.g. 99.5)')
	parser.add_argument('--max-detections', type=int, default=50,
						help='Max detections to keep after filtering & sorting')
	parser.add_argument('--conf', type=float, default=None,
						help='(Deprecated) alias for --score-threshold')
	parser.add_argument('--use-raw-scores', action='store_true',
						help='Flag reserved for future score transformations (currently no-op).')
	parser.add_argument('--pure-np', action='store_true',
						help='Force a pure numpy ONNXRuntime path (still uses torch for postprocess).')
	parser.add_argument('--direct-onnx', action='store_true',
						help='Bypass Infer wrapper and call onnxruntime session directly (optional).')
	parser.add_argument('--debug', action='store_true', help='Print extended diagnostics.')
	return parser.parse_args()


def decide_threshold(scores: torch.Tensor, args) -> float:
	"""Decide effective threshold based on args and available scores.

	Precedence:
	  1. --score-threshold or --conf (explicit)
	  2. --score-threshold-percentile
	  3. Fallback default 0.3
	"""
	explicit = args.score_threshold if args.score_threshold is not None else args.conf
	if explicit is not None:
		return float(explicit)
	if args.score_threshold_percentile is not None and scores.numel() > 0:
		p = float(args.score_threshold_percentile)
		p = max(0.0, min(100.0, p))
		thr = float(torch.quantile(scores.sort().values, p / 100.0).item())
		return thr
	return 0.3


def format_results(bboxes: torch.Tensor,
				   scores: torch.Tensor,
				   labels: torch.Tensor,
				   conf_thre: float,
				   max_det: int,
				   legacy_single_class: bool = True) -> List[Tuple[int, float, List[float]]]:
	"""Return filtered & sorted detection tuples.

	Each tuple: (raw_index, score, [x1,y1,x2,y2])
	Background (label != 0) filtered if legacy_single_class.
	"""
	dets: List[Tuple[int, float, List[float]]] = []
	if bboxes.numel() == 0:
		return dets
	for i in range(bboxes.shape[0]):
		s = float(scores[i].item())
		if s < conf_thre:
			continue
		lbl = int(labels[i].item())
		if legacy_single_class and lbl != 0:
			continue
		x1, y1, x2, y2 = [float(v) for v in bboxes[i].tolist()]
		dets.append((i, s, [x1, y1, x2, y2]))
	# Sort by descending score
	dets.sort(key=lambda t: t[1], reverse=True)
	if max_det and len(dets) > max_det:
		dets = dets[:max_det]
	return dets


def debug_print(scores: torch.Tensor, labels: torch.Tensor, conf: float):
	if scores.numel() == 0:
		print('[DEBUG] No boxes returned from model postprocess.')
		return
	max_score = scores.max().item()
	min_score = scores.min().item()
	above = (scores >= conf).sum().item()
	unique_labels = torch.unique(labels).cpu().tolist()
	print(f"[DEBUG] scores stats: count={scores.numel()} min={min_score:.4f} max={max_score:.4f} >=thr({conf:.4f})={above}")
	print(f"[DEBUG] unique raw labels: {unique_labels}")
	topk = min(10, scores.numel())
	vals, idxs = torch.topk(scores, k=topk)
	for rank, (v, idx) in enumerate(zip(vals, idxs)):
		print(f"[DEBUG] top{rank+1}: score={v.item():.4f} label={int(labels[idx])}")


def run_inference(args):
	if args.device == 'auto':
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	else:
		device = args.device

	if not os.path.isfile(args.onnx):
		raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
	if not os.path.isfile(args.image):
		raise FileNotFoundError(f"Image not found: {args.image}")
	os.makedirs(args.output, exist_ok=True)

	cfg = build_person_config(args.config)

	origin_img = np.asarray(Image.open(args.image).convert('RGB'))

	if args.direct_onnx:
		# Direct onnxruntime path
		import onnxruntime as ort
		if args.pure_np:
			print('[INFO] --pure-np + --direct-onnx: running direct ONNXRuntime session (postprocess still torch).')
		providers = None
		if device == 'cuda':
			# Attempt CUDA provider first
			available = ort.get_available_providers()
			if 'CUDAExecutionProvider' in available:
				providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
		session = ort.InferenceSession(args.onnx, providers=providers or ort.get_available_providers())
		input_name = session.get_inputs()[0].name
		# Preprocess (replicating Infer.preprocess logic)
		img_list = transform_img(origin_img, 0, **cfg.test.augment.transform, infer_size=args.infer_size)
		oh, ow, _ = origin_img.shape
		tensors = img_list.tensors  # shape [1,C,H,W]
		# Pad if smaller than target
		target_h, target_w = args.infer_size
		n, c, h, w = tensors.shape
		if h > target_h or w > target_w:
			raise ValueError(f"Image resized to ({h},{w}) exceeds target infer size {args.infer_size}")
		if (h, w) != (target_h, target_w):
			pad = torch.zeros(n, c, target_h, target_w, dtype=tensors.dtype)
			pad[:, :, :h, :w] = tensors
		else:
			pad = tensors
		image_list = ImageList(pad, [tensors.shape[-2:]], [pad.shape[-2:]])
		input_np = image_list.tensors.cpu().numpy()
		ort_outputs = session.run(None, {input_name: input_np})
		# Expect [scores, boxes]
		scores_t = torch.as_tensor(ort_outputs[0])  # [1,N,C]
		boxes_t = torch.as_tensor(ort_outputs[1])   # [1,N,4]
		processed = postprocess(scores_t, boxes_t,
								cfg.model.head.num_classes,
								cfg.model.head.nms_conf_thre,
								cfg.model.head.nms_iou_thre,
								image_list)
		output = processed[0].resize((ow, oh))
		bboxes = output.bbox
		scores = output.get_field('scores')
		cls_inds = output.get_field('labels')
		infer_engine = None  # We won't use visualization from wrapper unless we recreate vis call
	else:
		# Wrapper path (already uses onnxruntime internally)
		if args.pure_np:
			print('[INFO] pure-np flagged: using ONNXRuntime through Infer wrapper (postprocess remains torch-based).')
		infer_engine = Infer(cfg, infer_size=args.infer_size, device=device, ckpt=args.onnx, output_dir=args.output)
		bboxes, scores, cls_inds = infer_engine.forward(origin_img)

	# Determine threshold
	thr = decide_threshold(scores, args)

	if args.debug:
		debug_print(scores, cls_inds, thr)

	dets = format_results(bboxes, scores, cls_inds, thr, args.max_detections, legacy_single_class=True)

	# Fallback: if no detections pass threshold but we have scores, take top-K ignoring threshold
	if not dets and scores.numel() > 0:
		topk = min(args.max_detections, scores.numel())
		vals, idxs = torch.topk(scores, k=topk)
		fallback = []
		for s, idx in zip(vals, idxs):
			lbl = int(cls_inds[idx].item())
			if lbl != 0:  # still skip background
				continue
			x1, y1, x2, y2 = [float(v) for v in bboxes[idx].tolist()]
			fallback.append((int(idx), float(s.item()), [x1, y1, x2, y2]))
		dets = fallback
		if dets:
			print(f"[INFO] No boxes above threshold {thr:.4f}; using top-{len(dets)} fallback detections.")

	if not dets:
		print(f"No person detections (threshold={thr:.4f}).")
	else:
		print(f"Detections (threshold={thr:.4f}, kept={len(dets)}):")
		for rank, (raw_idx, score, (x1, y1, x2, y2)) in enumerate(dets, 1):
			print(f"  {rank:02d}. person (raw_idx={raw_idx}) score={score:.4f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

	# Visualization: reuse thr so displayed boxes align with what we printed
	save_name = os.path.basename(args.image)
	if infer_engine is not None:
		infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=thr, save_name=save_name, save_result=True)
	else:
		# Manual visualization using same util if wrapper not used
		from damo.utils.visualize import vis  # local import to avoid overhead when not needed
		vis_img = vis(origin_img.copy(), bboxes, scores, cls_inds, conf=thr, class_names=['person'])
		out_path = os.path.join(args.output, save_name)
		import cv2
		cv2.imwrite(out_path, vis_img[:, :, ::-1])

	return thr, dets


def main():
	args = parse_args()
	thr, dets = run_inference(args)
	if args.debug:
		print(f"[DEBUG] Finished inference. Effective threshold={thr:.4f} final_dets={len(dets)}")


if __name__ == '__main__':
	main()
