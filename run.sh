EXE_NAME="fast-detect-gpt"
MODEL_NAME="Qwen2.5-Coder-7B-Instruct-Q3_K_M.gguf"
INPUT_TEXT="Try to detect this :)"

MODEL_DIR="models"
BUILD_DIR="cmake-build-debug"

EXECUTABLE="./$BUILD_DIR/$EXE_NAME"
MODEL_PATH="./$MODEL_DIR/$MODEL_NAME"

$EXECUTABLE "$MODEL_PATH" "$INPUT_TEXT"