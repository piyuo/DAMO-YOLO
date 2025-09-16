#!/usr/bin/env python3
"""Pure ONNXRuntime single-class (person) inference script.

Refactored to remove any dependency on torch or the generic Infer wrapper.
All preprocessing, postprocessing, NMS, and visualization are handled with
NumPy, OpenCV, and ONNXRuntime only (plus config parsing for thresholds).

Example:
    python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
        --onnx pipeline/DAMO-YOLO/input/damoyolo_tinynasL25_S_person.onnx \
        --image pipeline/dataset/demo/demo.jpg \
        --output pipeline/output \
        --score-threshold 0.5

You can also use percentile-based adaptive thresholding:
    --score-threshold-percentile 98

If the exported ONNX uses a legacy head (person + background), the second
channel (background) is discarded so only the person channel is used.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

# Script location: repo_root/pipeline/DAMO-YOLO -> go two levels up to reach root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from damo.config.base import parse_config  # noqa: E402
from damo.utils.visualize import vis  # noqa: E402


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
						help='(No-op placeholder; raw scores already used).')
	parser.add_argument('--internal-nms-thr', type=float, default=None,
						help='Override NMS IoU threshold from config (optional).')
	parser.add_argument('--internal-pre-score', type=float, default=0.01,
						help='Score threshold used before NMS (on raw person scores).')
	parser.add_argument('--providers', default='',
						help="Comma-separated ONNX Runtime providers to force order (e.g. 'CPUExecutionProvider' or 'CUDAExecutionProvider,CPUExecutionProvider'). Leave empty for ORT defaults.")
	parser.add_argument('--debug', action='store_true', help='Print extended diagnostics.')
	return parser.parse_args()


def decide_threshold(scores: np.ndarray, args) -> float:
	"""Decide effective threshold based on args and available scores.

	Precedence:
	  1. --score-threshold or --conf (explicit)
	  2. --score-threshold-percentile
	  3. Fallback default 0.3
	"""
	explicit = args.score_threshold if args.score_threshold is not None else args.conf
	if explicit is not None:
		return float(explicit)
	if args.score_threshold_percentile is not None and scores.size > 0:
		p = float(args.score_threshold_percentile)
		p = max(0.0, min(100.0, p))
		thr = float(np.quantile(np.sort(scores), p / 100.0))
		return thr
	return 0.3


def format_results(bboxes: np.ndarray,
			   scores: np.ndarray,
			   labels: np.ndarray,
			   conf_thre: float,
			   max_det: int,
			   legacy_single_class: bool = True) -> List[Tuple[int, float, List[float]]]:
	"""Return filtered & sorted detection tuples.

	Each tuple: (raw_index, score, [x1,y1,x2,y2])
	Background (label != 0) filtered if legacy_single_class.
	"""
	dets: List[Tuple[int, float, List[float]]] = []
	if bboxes.size == 0:
		return dets
	for i in range(bboxes.shape[0]):
		s = float(scores[i])
		if s < conf_thre:
			continue
		lbl = int(labels[i])
		if legacy_single_class and lbl != 0:
			continue
		x1, y1, x2, y2 = [float(v) for v in bboxes[i].tolist()]
		dets.append((i, s, [x1, y1, x2, y2]))
	# Sort by descending score
	dets.sort(key=lambda t: t[1], reverse=True)
	if max_det and len(dets) > max_det:
		dets = dets[:max_det]
	return dets



def debug_print(scores: np.ndarray, labels: np.ndarray, conf: float):
	if scores.size == 0:
		print('[DEBUG] No boxes returned from model postprocess.')
		return
	max_score = float(scores.max())
	min_score = float(scores.min())
	above = int((scores >= conf).sum())
	unique_labels = sorted(set(labels.tolist()))
	print(f"[DEBUG] scores stats: count={scores.size} min={min_score:.4f} max={max_score:.4f} >=thr({conf:.4f})={above}")
	print(f"[DEBUG] unique raw labels: {unique_labels}")
	topk = min(10, scores.size)
	idxs = np.argsort(-scores)[:topk]
	for rank, idx in enumerate(idxs):
		print(f"[DEBUG] top{rank+1}: score={scores[idx]:.4f} label={int(labels[idx])}")


# ------------------------
# Minimal NumPy NMS helpers
# ------------------------
def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
		inds = np.where(ovr <= iou_thr)[0]
		order = order[inds + 1]
	return keep


def multiclass_nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, score_thr: float) -> np.ndarray | None:
	"""Return dets [K,6] -> x1,y1,x2,y2,score,cls (NumPy)."""
	final = []
	num_classes = scores.shape[1]
	for c in range(num_classes):
		cls_scores = scores[:, c]
		mask = cls_scores > score_thr
		if not mask.any():
			continue
		cls_boxes = boxes[mask]
		cls_scores_sel = cls_scores[mask]
		keep = _nms_single_class(cls_boxes, cls_scores_sel, iou_thr)
		if keep:
			kept_boxes = cls_boxes[keep]
			kept_scores = cls_scores_sel[keep]
			cls_ids = np.full((len(keep), 1), c)
			dets = np.concatenate([kept_boxes, kept_scores[:, None], cls_ids], axis=1)
			final.append(dets)
	if not final:
		return None
	return np.concatenate(final, axis=0)


def run_inference(args):
	if not os.path.isfile(args.onnx):
		raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
	if not os.path.isfile(args.image):
		raise FileNotFoundError(f"Image not found: {args.image}")
	os.makedirs(args.output, exist_ok=True)

	cfg = build_person_config(args.config)

	# Determine providers (user override > cuda preference > ORT defaults)
	custom_providers = [p.strip() for p in args.providers.split(',') if p.strip()]
	if custom_providers:
		provider_list = custom_providers
	else:
		provider_list = None
	if provider_list is None and args.device == 'cuda':
		available = ort.get_available_providers()
		if 'CUDAExecutionProvider' in available:
			provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
	session = ort.InferenceSession(args.onnx, providers=provider_list or ort.get_available_providers())
	input_meta = session.get_inputs()[0]
	input_name = input_meta.name

	# Load & preprocess image
	origin_img = np.asarray(Image.open(args.image).convert('RGB'))
	oh, ow, _ = origin_img.shape
	target_h, target_w = args.infer_size
	resized = cv2.resize(origin_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # (w,h)
	# Normalize (mean=0, std=1) -> no-op but keep structure if future changes
	img_chw = resized.transpose(2, 0, 1).astype(np.float32)
	batch = img_chw[None, ...]  # (1,C,H,W)

	# Run session
	outputs = session.run(None, {input_name: batch})
	# Expect [cls_scores, bboxes] shapes: (1,N,C) and (1,N,4)
	raw_scores = outputs[0]
	raw_boxes = outputs[1]
	if raw_scores.ndim != 3 or raw_boxes.ndim != 3:
		raise ValueError(f"Unexpected ONNX output shapes: {raw_scores.shape}, {raw_boxes.shape}")
	_, N, C = raw_scores.shape
	# Handle legacy background channel (keep only first class when single-class scenario)
	if C > 1:
		person_scores = raw_scores[0, :, 0:1]  # keep only person class
	else:
		person_scores = raw_scores[0]
	boxes = raw_boxes[0]  # (N,4)

	# Internal NMS / filtering
	score_thr_internal = args.internal_pre_score
	iou_thr = args.internal_nms_thr if args.internal_nms_thr is not None else getattr(cfg.model.head, 'nms_iou_thre', 0.7)
	dets = multiclass_nms_np(boxes, person_scores, iou_thr=iou_thr, score_thr=score_thr_internal)
	if dets is None:
		final_boxes = np.zeros((0, 4), dtype=np.float32)
		final_scores = np.zeros((0,), dtype=np.float32)
		final_labels = np.zeros((0,), dtype=np.int32)
	else:
		# dets: x1,y1,x2,y2,score,cls  (coordinates in resized image space)
		# Scale back to original image size (linear scaling since keep_ratio=False)
		scale_x = ow / float(target_w)
		scale_y = oh / float(target_h)
		final_boxes = dets[:, :4].copy()
		final_boxes[:, 0] *= scale_x
		final_boxes[:, 2] *= scale_x
		final_boxes[:, 1] *= scale_y
		final_boxes[:, 3] *= scale_y
		final_scores = dets[:, 4].astype(np.float32)
		final_labels = dets[:, 5].astype(np.int32)

	# Decide final display/printing threshold based on resulting scores
	thr = decide_threshold(final_scores, args)
	if args.debug:
		debug_print(final_scores, final_labels, thr)

	formatted = format_results(final_boxes, final_scores, final_labels, thr, args.max_detections, legacy_single_class=True)

	# Fallback if nothing above thr
	if not formatted and final_scores.size > 0:
		topk = min(args.max_detections, final_scores.size)
		idxs = np.argsort(-final_scores)[:topk]
		for idx in idxs:
			if final_labels[idx] != 0:
				continue
			x1, y1, x2, y2 = final_boxes[idx].tolist()
			formatted.append((int(idx), float(final_scores[idx]), [x1, y1, x2, y2]))
		if formatted:
			print(f"[INFO] No boxes above threshold {thr:.4f}; using top-{len(formatted)} fallback detections.")

	if not formatted:
		print(f"No person detections (threshold={thr:.4f}).")
	else:
		print(f"Detections (threshold={thr:.4f}, kept={len(formatted)}):")
		for rank, (raw_idx, score, (x1, y1, x2, y2)) in enumerate(formatted, 1):
			print(f"  {rank:02d}. person (raw_idx={raw_idx}) score={score:.4f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

	# Visualization with threshold thr
	vis_img = vis(origin_img.copy(), final_boxes, final_scores, final_labels, conf=thr, class_names=['person'])
	save_name = os.path.basename(args.image)
	out_path = os.path.join(args.output, save_name)
	cv2.imwrite(out_path, vis_img[:, :, ::-1])
	return thr, formatted


def main():
    args = parse_args()
    thr, dets = run_inference(args)
    if args.debug:
        print(f"[DEBUG] Finished inference. Effective threshold={thr:.4f} final_dets={len(dets)}")


if __name__ == '__main__':
    main()
