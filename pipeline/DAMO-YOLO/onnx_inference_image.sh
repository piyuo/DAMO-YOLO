# pipeline/DAMO-YOLO/onnx_inference_image.sh
#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
  --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
  --image pipeline/dataset/demo/demo.jpg \
  --output pipeline/output \
  --conf 0.3 \
  --pure-np \
  --no-vis