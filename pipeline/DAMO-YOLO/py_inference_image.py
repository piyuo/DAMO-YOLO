# pipeline/DAMO-YOLO/py_inference_image.py

#!/usr/bin/env python3
"""Simple script to run person-only DAMO-YOLO inference.

Uses the checkpoint at:
    pipeline/DAMO-YOLO/input/damoyolo_tinynasL25_S_person.pt
on the demo image:
    pipeline/dataset/demo/demo.jpg

Outputs an annotated image and prints raw detection results.
"""

import os
import sys
import argparse
from typing import List

import torch
import numpy as np
from PIL import Image

# Allow importing project modules when executed directly
# This script lives in repo_root/pipeline/DAMO-YOLO, so go up two levels to reach repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from damo.config.base import parse_config  # noqa: E402
from tools.demo import Infer  # noqa: E402


def build_person_config(config_path: str):
    """Load base config and adapt it for 1-class (person) inference.

    We modify:
      - head.num_classes -> 1
      - dataset.class_names -> ['person']
      - test.augment.transform.image_max_range -> (640, 640) (ensure expected size)
    """
    cfg = parse_config(config_path)
    # Treat checkpoint as a legacy single-class (person) model: legacy=True gives cls_out = num_classes + 1.
    cfg.model.head.num_classes = 1
    # Force legacy True (if key exists) so cls_out_channels matches checkpoint (2 channels)
    try:
        cfg.model.head.legacy = True
    except Exception:
        pass
    cfg.dataset.class_names = ['person']
    # Ensure test transform size is 640x640 (same as checkpoint training)
    if hasattr(cfg.test, 'augment') and hasattr(cfg.test.augment, 'transform'):
        cfg.test.augment.transform.image_max_range = (640, 640)
    return cfg


def format_results(bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, conf_thre: float, legacy_single_class: bool = True) -> List[str]:
    """Format detections.

    legacy_single_class: if True, interpret label 0 as person and (if present) label 1 as background.
    In legacy single-class mode (num_classes=1, legacy=True):
        cls_out_channels = 2, and the last channel is background (index 1), so label 0 = person.
    The current multiclass_nms implementation may still emit background boxes if background scores pass threshold; we filter them out.
    """
    lines: List[str] = []
    if bboxes.numel() == 0:
        return lines
    for i in range(bboxes.shape[0]):
        score = scores[i].item()
        if score < conf_thre:
            continue
        x1, y1, x2, y2 = bboxes[i].tolist()
        raw_label = int(labels[i].item())
        if legacy_single_class:
            # Background is expected to be last index (==1); only keep raw_label == 0
            if raw_label != 0:
                continue
        # For single-class output, label is always person
        lines.append(
            f"person bbox[{i}] (class_id={raw_label}): (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}), score={score:.3f}")
    return lines


def parse_args():
    parser = argparse.ArgumentParser("Person inference")
    parser.add_argument('--config', default=os.path.join(ROOT, 'configs', 'damoyolo_tinynasL25_S.py'))
    parser.add_argument('--ckpt', default=os.path.join(ROOT, 'pipeline', 'DAMO-YOLO', 'input', 'damoyolo_tinynasL25_S_person.pt'))
    parser.add_argument('--image', default=os.path.join(ROOT, 'pipeline', 'dataset', 'demo', 'demo.jpg'))
    parser.add_argument('--output', default=os.path.join(ROOT, 'pipeline', 'output'))
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold for printing + visualization filtering')
    parser.add_argument('--infer-size', type=int, nargs=2, default=[640, 640], help='inference size (h w)')
    parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--debug', action='store_true', help='print extra score/label diagnostics')
    return parser.parse_args()


def debug_print(scores: torch.Tensor, labels: torch.Tensor, conf: float):
    if scores.numel() == 0:
        print('[DEBUG] No boxes returned from model.')
        return
    max_score = scores.max().item()
    min_score = scores.min().item()
    above = (scores >= conf).sum().item()
    unique_labels = torch.unique(labels).cpu().tolist()
    print(f"[DEBUG] scores: count={scores.numel()} min={min_score:.3f} max={max_score:.3f} >=conf({conf})={above}")
    print(f"[DEBUG] unique raw labels (first 20 shown): {unique_labels[:20]}")
    # Show top 10 by score
    topk = min(10, scores.numel())
    vals, idxs = torch.topk(scores, k=topk)
    for rank, (v, idx) in enumerate(zip(vals, idxs)):
        print(f"[DEBUG] top{rank+1}: score={v.item():.3f} label={int(labels[idx])}")


def main():
    args = parse_args()

    config_file = args.config
    ckpt_path = args.ckpt
    image_path = args.image
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    cfg = build_person_config(config_file)
    infer_engine = Infer(cfg, infer_size=args.infer_size, device=device, ckpt=ckpt_path, output_dir=output_dir)

    origin_img = np.asarray(Image.open(image_path).convert('RGB'))
    bboxes, scores, cls_inds = infer_engine.forward(origin_img)

    if args.debug:
        debug_print(scores, cls_inds, args.conf)

    # We now treat the model as legacy single-class person detector
    lines = format_results(bboxes, scores, cls_inds, args.conf, legacy_single_class=True)
    if not lines:
        print(f"No person detections above conf {args.conf}")
    else:
        print("Detections (filtered):")
        for l in lines:
            print(l)

    save_name = os.path.basename(image_path)
    infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=args.conf, save_name=save_name, save_result=True)
    #print(f"Annotated image saved to: {os.path.join(output_dir, save_name)}")


if __name__ == '__main__':
    main()
