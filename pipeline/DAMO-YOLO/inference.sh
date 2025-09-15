# pipeline/DAMO-YOLO/inference.sh

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

python pipeline/DAMO-YOLO/inference.py --debug
