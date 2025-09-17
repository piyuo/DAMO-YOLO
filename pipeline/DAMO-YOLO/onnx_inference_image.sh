# pipeline/DAMO-YOLO/onnx_inference_image.sh
#!/bin/bash
# pipeline/DAMO-YOLO/onnx_inference_image.sh

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

echo "🔧 Running ONNX inference..."

python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
  --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
  --image pipeline/dataset/demo/demo.jpg \
  --output pipeline/output \
  --use-raw-scores \
  --score-threshold 0.25\
  --max-detections 12 \
  --debug

echo "✅ Fixed ONNX inference completed! Check pipeline/output/ for result images."