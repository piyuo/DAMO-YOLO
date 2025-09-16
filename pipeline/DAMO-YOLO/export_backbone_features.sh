# pipeline/DAMO-YOLO/export_backbone_features.sh
#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

python3 pipeline/DAMO-YOLO/export_backbone_features.py \
--onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
--select-nodes /neck/merge_5/Concat_output_0,/neck/merge_7/Concat_output_0,/neck/merge_6/Concat_output_0 \
--patch --dump-selected-shapes \
--image pipeline/dataset/demo/demo.jpg \
--export-features --force-cpu --verbose
#--summarize




#/neck/merge_5/Concat_output_0 (P3, 80x80)
#/neck/merge_7/Concat_output_0 (P4, 40x40)
#/neck/merge_6/Concat_output_0 (P5, 20x20)