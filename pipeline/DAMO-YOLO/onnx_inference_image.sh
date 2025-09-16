# pipeline/DAMO-YOLO/onnx_inference_image.sh
#!/bin/bash
# pipeline/DAMO-YOLO/onnx_inference_image_fixed.sh
# Fixed version of ONNX inference script that handles low confidence scores properly
# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

echo "🔧 Running fixed ONNX inference..."

python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
  --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
  --image pipeline/dataset/demo/demo.jpg \
  --output pipeline/output \
  --use-raw-scores \
  --score-threshold 0.35\
  --max-detections 12 \
  --pure-np \
  --debug

echo "✅ Fixed ONNX inference completed! Check pipeline/output/ for result images."