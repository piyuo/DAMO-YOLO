# pipeline/DAMO-YOLO/py_inference_image.sh

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

python pipeline/DAMO-YOLO/inference_image.py  --debug --image pipeline/dataset/demo/3.png
