# pipeline/DAMO-YOLO/py_inference_image.sh

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

python3 pipeline/DAMO-YOLO/py_inference_image.py  --debug --image pipeline/dataset/demo/demo.jpg
