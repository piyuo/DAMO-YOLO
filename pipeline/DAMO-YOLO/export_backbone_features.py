#!/usr/bin/env python3
"""Export backbone feature maps from a DAMO-YOLO ONNX model.

Goal:
  1. Optionally patch an existing ONNX model to expose internal backbone feature
     map tensors (multi‑scale feature pyramid inputs) as new graph outputs.
  2. Run inference on an image (or a directory of images) and save the selected
     feature maps as .npy files under a target directory (default:
     pipeline/DAMO-YOLO/backbone).

Why needed:
  BOT-SORT (and other trackers) may require per-frame feature maps / embeddings.
  This script gives you direct access to the feature pyramid outputs so you can
  build appearance descriptors or feed them into downstream modules.

Key ideas:
  - Standard exported DAMO-YOLO ONNX usually outputs only classification scores
    & bounding boxes. Intermediate backbone/neck feature maps are *not* graph
    outputs, so ONNXRuntime cannot return them directly.
  - We load the ONNX graph, run shape inference, heuristically identify likely
    multi-scale feature maps (three or more 4D tensors with descending spatial
    sizes, e.g. 80x80, 40x40, 20x20 for 640 input) and append them as new
    outputs.
  - You can also list all candidate 4D tensors and manually choose names.

Workflow examples:
  1. Just list candidate internal tensors (no modifications):
       python pipeline/DAMO-YOLO/export_backbone_features.py \
         --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx --list-nodes

  2. Auto-select 3 backbone feature maps, create a patched ONNX, and export
     features for a single image:
       python pipeline/DAMO-YOLO/export_backbone_features.py \
         --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
         --image pipeline/dataset/demo/demo.jpg \
         --auto-select --patch --export-features

  3. Manually pick node names (comma separated) to expose & export for a folder:
       python pipeline/DAMO-YOLO/export_backbone_features.py \
         --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
         --image-dir path/to/frames \
         --select-nodes "/model/backbone/stage3_out,/model/backbone/stage4_out,/model/backbone/stage5_out" \
         --patch --export-features

Caveats:
  - Heuristic auto-selection may mis-pick if graph structure differs; always
    verify shapes match expectations (descending spatial dims, channels like
    256/512/1024 etc.). Use --list-nodes first if unsure.
  - If shape inference fails to determine all tensor shapes, you may need to
    install a newer onnx version, or fall back to manual node selection.
  - This script does not compute ReID embeddings; it only dumps raw feature
    maps. For BOT-SORT you typically pool/normalize or pass through a separate
    embedding head.

Author: Auto-generated helper script.
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Tuple, Dict, Sequence

import numpy as np
from PIL import Image
import cv2
import onnx
import onnx.shape_inference
import onnxruntime as ort

# Repo root (script lives at pipeline/DAMO-YOLO/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'backbone')


def parse_args():
    p = argparse.ArgumentParser("Export backbone feature maps from ONNX")
    p.add_argument('--onnx', required=True, help='Path to original ONNX model')
    p.add_argument('--patched-onnx', default=None,
                   help='(Optional) Output path for patched ONNX (defaults to <orig_basename>_feat.onnx in same dir)')
    p.add_argument('--image', default=None, help='Single image path to run inference')
    p.add_argument('--image-dir', default=None, help='Directory of images (jpg/png) for batch export')
    p.add_argument('--infer-size', type=int, nargs=2, default=[640, 640], help='Inference size (H W) to resize images (no keep ratio)')
    p.add_argument('--providers', default='', help='Comma separated ORT providers (override).')
    p.add_argument('--force-cpu', action='store_true', help='Force CPUExecutionProvider only (helpful on macOS to avoid CoreML shape/output issues).')
    p.add_argument('--list-nodes', action='store_true', help='List candidate 4D internal tensors & exit.')
    p.add_argument('--summarize', action='store_true', help='Print grouped summary of candidate tensors by spatial size (without early exit).')
    p.add_argument('--auto-select', action='store_true', help='Automatically pick likely 3 multi-scale feature maps.')
    p.add_argument('--select-nodes', default='', help='Comma-separated exact tensor names to expose as outputs.')
    p.add_argument('--patch', action='store_true', help='Create patched ONNX with selected feature nodes exposed.')
    p.add_argument('--export-features', action='store_true', help='Run inference and save chosen feature maps (.npy).')
    p.add_argument('--save-dir', default=DEFAULT_SAVE_DIR, help='Directory to save feature .npy files.')
    p.add_argument('--verbose', action='store_true', help='Verbose logging.')
    p.add_argument('--dump-selected-shapes', action='store_true', help='After (patched) inference on a single image, print shapes of selected feature outputs (implies --export-features for that image, but will not save .npy unless --export-features is also given).')
    p.add_argument('--alias-names', default='', help='Comma-separated short names (e.g. p3,p4,p5) to add as Identity outputs for the selected nodes.')
    p.add_argument('--drop-original-outputs', action='store_true', help='When alias-names are provided, remove the originally added outputs (effectively renaming instead of duplicating).')
    return p.parse_args()


# ----------------------------- ONNX Graph Utilities -----------------------------

def load_and_infer(model_path: str) -> onnx.ModelProto:
    model = onnx.load(model_path)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        return inferred
    except Exception as e:
        print(f"[WARN] Shape inference failed ({e}); proceeding with raw model.")
        return model


def extract_4d_candidates(model: onnx.ModelProto) -> List[Tuple[str, List[int]]]:
    """Return list of (tensor_name, dims) for 4D tensors (N,C,H,W) inside value_info.
    Excludes graph inputs & already-declared outputs.
    """
    graph = model.graph
    output_names = {o.name for o in graph.output}
    input_names = {i.name for i in graph.input}
    candidates = []
    # Collect value_info entries (intermediate tensors) + outputs (some may be helpful)
    for vi in list(graph.value_info) + list(graph.output):
        name = vi.name
        if name in input_names:
            continue
        t = vi.type
        if not t.HasField('tensor_type'):
            continue
        shp = t.tensor_type.shape
        dims: List[int] = []
        concrete = True
        for d in shp.dim:
            if d.HasField('dim_value'):
                dims.append(d.dim_value)
            else:
                dims.append(-1)
                concrete = False
        if len(dims) == 4:
            # Basic sanity: batch 1 or unknown, spatial positive if known
            if (dims[0] in (1, -1)) and (dims[2] != 1 or dims[3] != 1):
                candidates.append((name, dims))
    return candidates


def auto_select_features(candidates: List[Tuple[str, List[int]]], k: int = 3) -> List[str]:
    """Heuristically choose k multi-scale feature maps with descending spatial resolution.

    Improved strategy:
      1. Keep only tensors with H,W >= 8 (ignore very small post-head tensors) and H,W not 1.
      2. Group by (H,W); pick representative with largest channel count per group.
      3. Sort groups by H descending.
      4. Walk groups and require roughly pyramidal pattern: each next H about half of previous (ratio between 1.6 and 2.6 tolerated) and strictly smaller.
      5. Prefer channel counts that are non-decreasing as spatial size shrinks (typical backbone/neck pattern).
    If strict pattern not found, fallback to previous simple top-k distinct spatial sizes.
    """
    groups: Dict[Tuple[int,int], List[Tuple[str,List[int]]]] = {}
    for name, dims in candidates:
        N,C,H,W = dims
        if H <= 1 or W <= 1:
            continue
        if H < 8 or W < 8:  # ignore tiny maps (likely post-processing or head internals)
            continue
        groups.setdefault((H,W), []).append((name,dims))
    reps: List[Tuple[str,List[int]]] = []
    for (H,W), lst in groups.items():
        # choose largest C as representative
        lst_sorted = sorted(lst, key=lambda x: x[1][1], reverse=True)
        reps.append(lst_sorted[0])
    if not reps:
        return []
    reps.sort(key=lambda x: x[1][2], reverse=True)  # by H desc

    # Fast path: prefer canonical YOLO pyramid (approx 80,40,20). Allow small tolerance.
    target_sizes = [80, 40, 20]
    size_to_rep: Dict[int, Tuple[str,List[int]]] = {dims[2]: (name,dims) for name,dims in reps}
    picked = []
    for t in target_sizes:
        # find closest size within tolerance 2
        best = None
        for h in size_to_rep.keys():
            if abs(h - t) <= 2:
                cand = size_to_rep[h]
                # ensure channel count grows or stays the same relative to previous (if any)
                if not picked or cand[1][1] >= picked[-1][1][1] * 0.8:
                    best = cand
                    break
        if best:
            picked.append(best)
    if len(picked) == k:
        return [n for n,_ in picked]

    def is_pyramid(prev_h: int, h: int) -> bool:
        if h >= prev_h:
            return False
        ratio = prev_h / float(h)
        return 1.6 <= ratio <= 2.6  # allow some deviation from exact 2

    # Try to build a pyramid
    chosen: List[Tuple[str,List[int]]] = []
    for cand in reps:
        if not chosen:
            chosen.append(cand)
            continue
        if len(chosen) >= k:
            break
        prev_h = chosen[-1][1][2]
        h = cand[1][2]
        # channel monotonicity preference
        prev_c = chosen[-1][1][1]
        c = cand[1][1]
        if is_pyramid(prev_h, h) and c >= prev_c * 0.5:  # allow slight drop but discourage random early layers
            chosen.append(cand)
    if len(chosen) < k:
        # Fallback: pick first k distinct spatial sizes
        fallback = []
        seen = set()
        for name,dims in reps:
            hw = (dims[2], dims[3])
            if hw in seen:
                continue
            fallback.append((name,dims))
            seen.add(hw)
            if len(fallback) >= k:
                break
        chosen = fallback
    return [n for n,_ in chosen]


def patch_model_with_outputs(model: onnx.ModelProto, tensor_names: Sequence[str], verbose=False) -> onnx.ModelProto:
    existing = {o.name for o in model.graph.output}
    value_info_map = {vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)}
    added = []
    for name in tensor_names:
        if name in existing:
            if verbose:
                print(f"[INFO] Tensor already an output: {name}")
            continue
        vi = value_info_map.get(name)
        if vi is None:
            # Create a minimal ValueInfo (shape unknown); ONNXRuntime can still produce it.
            from onnx import helper, TensorProto
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            if verbose:
                print(f"[WARN] Shape unknown for {name}; adding generic FLOAT output.")
        model.graph.output.extend([vi])
        added.append(name)
    if verbose and added:
        print(f"[INFO] Added {len(added)} new outputs: {added}")
    return model


def add_alias_outputs(model: onnx.ModelProto, original: Sequence[str], aliases: Sequence[str], drop_original: bool=False, verbose: bool=False) -> onnx.ModelProto:
    """Insert Identity nodes to alias existing tensor outputs.

    If drop_original is True, will remove the original graph.output entries whose
    names are in `original` after adding alias outputs. (Underlying internal tensor
    names remain; we only change exported names.)
    """
    from onnx import helper
    graph = model.graph
    existing_outputs = {o.name: o for o in graph.output}
    # Build map of value info for shape copying
    vi_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.output)}
    for orig, alias in zip(original, aliases):
        if alias in existing_outputs:
            if verbose:
                print(f"[WARN] Alias name {alias} already an output; skipping.")
            continue
        # Create identity node
        node = helper.make_node('Identity', inputs=[orig], outputs=[alias], name=f"Alias_{alias}")
        graph.node.append(node)
        # Copy shape info if available
        vi_src = vi_map.get(orig)
        if vi_src is not None:
            new_vi = helper.make_tensor_value_info(alias, vi_src.type.tensor_type.elem_type,
                                                   [d.dim_value if d.HasField('dim_value') else None for d in vi_src.type.tensor_type.shape.dim])
        else:
            from onnx import TensorProto
            new_vi = helper.make_tensor_value_info(alias, TensorProto.FLOAT, None)
        graph.output.append(new_vi)
        if verbose:
            print(f"[INFO] Added alias output {alias} -> {orig}")
    if drop_original:
        kept = []
        removed = []
        for o in graph.output:
            if o.name in original:
                removed.append(o.name)
            else:
                kept.append(o)
        if removed and verbose:
            print(f"[INFO] Dropping original outputs (renamed via aliases): {removed}")
        # Reassign outputs
        del graph.output[:]  # clear
        graph.output.extend(kept)
    return model


# ----------------------------- Inference & Export -----------------------------

def build_session(onnx_path: str, providers_arg: str, force_cpu: bool=False):
    custom = [p.strip() for p in providers_arg.split(',') if p.strip()]
    if force_cpu:
        provider_list = ['CPUExecutionProvider']
    elif custom:
        provider_list = custom
    else:
        provider_list = None  # ORT decides
    try:
        sess = ort.InferenceSession(onnx_path, providers=provider_list or ort.get_available_providers())
    except Exception as e:
        # Fallback to CPU if CoreML / GPU provider caused a failure after patching outputs
        print(f"[WARN] Provider initialization failed ({e}); retrying with CPUExecutionProvider only.")
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_info = sess.get_inputs()[0]
    return sess, input_info.name


def preprocess_image(path: str, infer_h: int, infer_w: int) -> Tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(path).convert('RGB'))
    resized = cv2.resize(img, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose(2,0,1).astype(np.float32)  # (C,H,W)
    return img, chw[None]


def export_features(sess: ort.InferenceSession, input_name: str, image_paths: List[str], feature_names: List[str], save_dir: str, infer_size: Tuple[int,int], verbose=False):
    os.makedirs(save_dir, exist_ok=True)
    h,w = infer_size
    exported = []
    for img_path in image_paths:
        orig, batch = preprocess_image(img_path, h, w)
        outputs = sess.run(None, {input_name: batch})
        # ORT returns outputs in graph output order; assume feature_names are subset and appear (likely at end if patched)
        name_to_arr = {}
        for meta, arr in zip(sess.get_outputs(), outputs):
            name_to_arr[meta.name] = arr
        base = os.path.splitext(os.path.basename(img_path))[0]
        for fname in feature_names:
            if fname not in name_to_arr:
                if verbose:
                    print(f"[WARN] Feature {fname} missing in outputs for {img_path}")
                continue
            feat = name_to_arr[fname]
            out_path = os.path.join(save_dir, f"{base}__{sanitize(fname)}__{feat.shape}.npy")
            np.save(out_path, feat)
            exported.append(out_path)
            if verbose:
                print(f"[OK] Saved {feat.shape} -> {out_path}")
    return exported


def sanitize(name: str) -> str:
    return name.replace('/', '_').replace('::','_').replace(':','_')


# ----------------------------- Main Logic -----------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)

    model = load_and_infer(args.onnx)
    candidates = extract_4d_candidates(model)

    if args.list_nodes:
        print(f"Found {len(candidates)} candidate 4D tensors (name, NCHW):")
        for name,dims in sorted(candidates, key=lambda x: (x[1][2], x[1][3]), reverse=True):
            print(f"  {name:60s} {dims}")
        print("(Use --select-nodes with comma-separated exact names to expose them.)")
        return

    if args.summarize:
        # Group by spatial size and print top few channel variants
        from collections import defaultdict
        groups = defaultdict(list)
        for name,dims in candidates:
            _,C,H,W = dims
            groups[(H,W)].append((name,C))
        print(f"[SUMMARY] Grouping {len(candidates)} tensors by spatial size (H,W):")
        for (H,W) in sorted(groups.keys(), reverse=True):
            lst = sorted(groups[(H,W)], key=lambda x: x[1], reverse=True)
            top = lst[:5]
            top_str = ', '.join([f"{n} (C={c})" for n,c in top])
            print(f"  ({H:>3},{W:>3}) -> {len(lst):>3} tensors. Top: {top_str}")
        print("[HINT] Backbone/FPN multi-scale features usually appear as three sizes with descending H,W (e.g. 80,40,20 for 640 input).")

    selected: List[str] = []
    if args.select_nodes:
        selected = [n.strip() for n in args.select_nodes.split(',') if n.strip()]
    elif args.auto_select:
        selected = auto_select_features(candidates, k=3)
        print(f"[INFO] Auto-selected feature tensors: {selected}")
    else:
        print("[ERROR] No feature selection specified. Use --auto-select or --select-nodes or --list-nodes.")
        sys.exit(2)

    if not selected:
        print("[ERROR] Empty selection after processing. Aborting.")
        sys.exit(3)

    patched_path = args.patched_onnx
    if args.patch:
        if patched_path is None:
            base, ext = os.path.splitext(args.onnx)
            patched_path = base + '_feat' + ext
        if os.path.exists(patched_path):
            print(f"[INFO] Overwriting existing patched model: {patched_path}")
        patched = patch_model_with_outputs(model, selected, verbose=args.verbose)
        # Handle alias names if requested
        if args.alias_names:
            alias_list = [a.strip() for a in args.alias_names.split(',') if a.strip()]
            if len(alias_list) != len(selected):
                print(f"[ERROR] alias-names count ({len(alias_list)}) != selected tensor count ({len(selected)}).")
                sys.exit(9)
            patched = add_alias_outputs(patched, selected, alias_list, drop_original=args.drop_original_outputs, verbose=args.verbose)
            # If originals were dropped, update the working selection list to the alias names
            if args.drop_original_outputs:
                if args.verbose:
                    print(f"[INFO] Replacing selected tensor list with aliases: {alias_list}")
                selected = alias_list
        onnx.save(patched, patched_path)
        print(f"[OK] Saved patched model with {len(selected)} extra outputs -> {patched_path}")
    else:
        # If not patching, we cannot expose new outputs unless they already exist.
        missing = [n for n in selected if n not in {o.name for o in model.graph.output}]
        if missing:
            print(f"[ERROR] Selected nodes are not current outputs: {missing}. Add --patch to expose them.")
            sys.exit(4)
        patched_path = args.onnx
        # If aliasing without patch requested, modify in-place
        if args.alias_names:
            alias_list = [a.strip() for a in args.alias_names.split(',') if a.strip()]
            if len(alias_list) != len(selected):
                print(f"[ERROR] alias-names count ({len(alias_list)}) != selected tensor count ({len(selected)}).")
                sys.exit(9)
            temp_model = onnx.load(patched_path)
            temp_model = add_alias_outputs(temp_model, selected, alias_list, drop_original=args.drop_original_outputs, verbose=args.verbose)
            onnx.save(temp_model, patched_path)
            if args.verbose:
                print(f"[INFO] Saved alias-augmented model -> {patched_path}")
            if args.drop_original_outputs:
                if args.verbose:
                    print(f"[INFO] Replacing selected tensor list with aliases: {alias_list}")
                selected = alias_list

    if args.export_features or args.dump_selected_shapes:
        # Build session on patched model
        sess, input_name = build_session(patched_path, args.providers, force_cpu=args.force_cpu)
        # Validate that selected outputs are present
        out_names = [o.name for o in sess.get_outputs()]
        for n in selected:
            if n not in out_names:
                print(f"[ERROR] Patched session missing expected output {n}")
                sys.exit(5)
        # Collect images
        image_paths: List[str] = []
        if args.image:
            if not os.path.isfile(args.image):
                print(f"[ERROR] Image not found: {args.image}")
                sys.exit(6)
            image_paths.append(args.image)
        if args.image_dir:
            if not os.path.isdir(args.image_dir):
                print(f"[ERROR] Image dir missing: {args.image_dir}")
                sys.exit(7)
            exts = ('*.jpg','*.jpeg','*.png','*.bmp')
            for e in exts:
                image_paths.extend(sorted(glob.glob(os.path.join(args.image_dir, e))))
        if not image_paths:
            print("[ERROR] No images specified. Provide --image or --image-dir when using --export-features.")
            sys.exit(8)
        exported = []
        if args.export_features:
            exported = export_features(sess, input_name, image_paths, selected, args.save_dir, tuple(args.infer_size), verbose=args.verbose)
            print(f"[DONE] Exported {len(exported)} feature maps from {len(image_paths)} images to {args.save_dir}")
        # Dump shapes (use first image only) if requested
        if args.dump_selected_shapes:
            if args.verbose:
                print("[INFO] Running single image inference for shape dump.")
            orig, batch = preprocess_image(image_paths[0], *args.infer_size)
            outputs = sess.run(None, {input_name: batch})
            meta = sess.get_outputs()
            name_to_shape = {m.name: outputs[i].shape for i,m in enumerate(meta)}
            shapes_report = "\n".join([f"  {n}: {name_to_shape.get(n)}" for n in selected])
            print(f"[SHAPES] Selected feature output shapes:\n{shapes_report}")
    else:
        print("[INFO] Skipped feature export (omit --export-features to only create patched model).")


if __name__ == '__main__':
    main()
