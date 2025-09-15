# pipeline/DAMO-YOLO/convert_damo_to_onnx.sh
#!/bin/bash

# ==============================================================================
# DAMO-YOLO PyTorch to ONNX Conversion Script
# ==============================================================================
# Description:
# This script automates the conversion of a DAMO-YOLO .pt model to the ONNX
# format. It performs the following steps:
#   1. Sets up a Python virtual environment to manage dependencies.
#   2. Installs required packages (torch, onnx, onnx-simplifier, etc.).
#   3. Installs the DAMO-YOLO project in editable mode.
#   4. Runs the official converter.py script with the specified parameters.
#   5. Moves the generated ONNX file to the final destination.
#
# Prerequisites:
#   - This script must be run from the root directory of the DAMO-YOLO repo.
#   - The PyTorch model (.pt) must be located at the specified INPUT_MODEL_PT path.
#   - The corresponding model config file must exist at the CONFIG_FILE path.
# ==============================================================================

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines fail if any command fails, not just the last one.
set -o pipefail

# --- User-Defined Variables ---
# Define paths and model parameters here for easy modification.

# The Python script used for conversion
CONVERTER_SCRIPT="tools/converter.py"

# The configuration file corresponding to the model architecture.
# IMPORTANT: Must be a class-based config (contains a Config class) for parse_config to work.
# Added a new single-class config at configs/damoyolo_tinynasL25_S_person.py
CONFIG_FILE="configs/damoyolo_tinynasL25_S_person.py"

# Path to the input PyTorch model checkpoint (.pt file)
INPUT_MODEL_PT="pipeline/DAMO-YOLO/input/damoyolo_tinynasL25_S_person.pt"

# Final desired path for the converted ONNX model
FINAL_OUTPUT_ONNX="pipeline/DAMO-YOLO/output/damoyolo_tinynasL25_S_person.onnx"

# The ONNX file name is derived automatically from the config file name inside converter.py.
# We'll compute it here too for clarity and moving.
GENERATED_ONNX_NAME="$(basename "$CONFIG_FILE" .py).onnx"

# Model inference parameters
IMAGE_SIZE=640
BATCH_SIZE=1

# Python virtual environment directory name
VENV_DIR=".venv"

# --- 1. Environment Setup ---
echo "▶️  Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅  Virtual environment created at '$VENV_DIR'."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment activated."

# --- 2. Install Dependencies ---
echo "▶️  Installing required Python packages..."
pip install --upgrade pip
# Allow user to pre-install torch with CUDA. If not present, install CPU version as fallback.
if python -c "import torch" 2>/dev/null; then
    echo "ℹ️  torch already installed, skipping explicit torch install."
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install onnx loguru termcolor easydict tabulate
# Try to install onnx-simplifier (onnxsim) but don't fail if it breaks
set +e
pip install onnxsim==0.4.33 || echo "⚠️  onnxsim install failed or skipped; continuing without simplification"
set -e
pip install --no-build-isolation -e . # Install DAMO-YOLO project dependencies (needs torch present)
echo "✅  All dependencies installed successfully."

# --- 3. Create Output Directory ---
# The -p flag ensures the directory is created only if it doesn't exist.
mkdir -p "$(dirname "$FINAL_OUTPUT_ONNX")"
echo "✅  Ensured output directory exists."

# --- 4. Run ONNX Conversion ---
echo "🚀  Starting model conversion to ONNX..."
echo "=========================================="

set +e
python "$CONVERTER_SCRIPT" \
        -f "$CONFIG_FILE" \
        -c "$INPUT_MODEL_PT" \
        --img_size "$IMAGE_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --opset 11
status=$?
set -e

if [ $status -ne 0 ]; then
    echo "❌  Conversion failed (exit code $status). Common causes:"
    echo "    - Bad config file (must define Config class)."
    echo "    - Missing checkpoint: $INPUT_MODEL_PT"
    echo "    - Shape or device issues."
    echo "Check the Python traceback above."
    deactivate || true
    exit $status
fi

echo "=========================================="
echo "✅  Conversion script finished."

# --- 5. Finalize and Clean Up ---
echo "▶️  Moving generated model to final destination..."

# Check if the generated ONNX file exists before moving
if [ -f "$GENERATED_ONNX_NAME" ]; then
    mv -f "$GENERATED_ONNX_NAME" "$FINAL_OUTPUT_ONNX"
    echo "✅  Successfully moved model to: $FINAL_OUTPUT_ONNX"
else
    echo "❌  ERROR: The converter script did not generate the expected ONNX file: $GENERATED_ONNX_NAME"
    exit 1
fi

# Deactivate the virtual environment
deactivate
echo "✅  Virtual environment deactivated."

echo "🎉  Process complete! Your ONNX model is ready."