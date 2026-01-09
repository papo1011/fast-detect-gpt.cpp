EXE_NAME="fast-detect-gpt"
BUILD_DIR="build"
INPUT_DIR="inputs"

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "'$ENV_FILE' not found. Please create it"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

EXECUTABLE="./$BUILD_DIR/$EXE_NAME"
INPUT_PATH="./$INPUT_DIR/$TRAIN_FILE"

$EXECUTABLE --find-threshold -f "$INPUT_PATH" --label-col label