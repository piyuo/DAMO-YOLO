# pipeline/DAMO-YOLO/inference.py

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
from typing import List

import torch
import numpy as np
from PIL import Image

# Allow importing project modules when executed directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
    # Adjust to 2-class (background + person) head to match checkpoint shapes
    # The checkpoint appears to have cls_out_channels=2.
    cfg.model.head.num_classes = 2
    cfg.dataset.class_names = ['__bg__', 'person']
    # Ensure test transform size is 640x640 (same as checkpoint training)
    if hasattr(cfg.test, 'augment') and hasattr(cfg.test.augment, 'transform'):
        cfg.test.augment.transform.image_max_range = (640, 640)
    return cfg


def format_results(bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, conf_thre: float) -> List[str]:
    lines = []
    for i in range(bboxes.shape[0]):
        score = scores[i].item()
        if score < conf_thre:
            continue
        x1, y1, x2, y2 = bboxes[i].tolist()
        cls_id = int(labels[i].item())
        if cls_id == 0:  # skip background
            continue
        lines.append(f"person bbox[{i}]: (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}), score={score:.3f}")
    return lines


def main():
    # Paths
    config_file = os.path.join(ROOT, 'configs', 'damoyolo_tinynasL25_S.py')
    ckpt_path = os.path.join(ROOT, 'pipeline', 'DAMO-YOLO', 'input', 'damoyolo_tinynasL25_S_person.pt')
    image_path = os.path.join(ROOT, 'pipeline', 'dataset', 'demo', 'demo.jpg')
    output_dir = os.path.join(ROOT, 'person_demo')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build config and inference engine
    cfg = build_person_config(config_file)
    infer_engine = Infer(cfg, infer_size=[640, 640], device=device, ckpt=ckpt_path, output_dir=output_dir)

    # Load image
    origin_img = np.asarray(Image.open(image_path).convert('RGB'))
    bboxes, scores, cls_inds = infer_engine.forward(origin_img)

    # Print results
    conf_thre = 0.25
    lines = format_results(bboxes, scores, cls_inds, conf_thre)
    if not lines:
        print(f"No person detections above conf {conf_thre}")
    else:
        print("Detections (filtered):")
        for l in lines:
            print(l)

    # Save visualization
    save_name = os.path.basename(image_path)
    infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=conf_thre, save_name=save_name, save_result=True)
    print(f"Annotated image saved to: {os.path.join(output_dir, save_name)}")


if __name__ == '__main__':
    main()
