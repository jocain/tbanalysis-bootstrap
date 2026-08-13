#!/bin/bash
INPUT_DIR=$1
OUTPUT_DIR=$2
PAIR=$3

echo "Setting up environment..."
source $WORK_DIR/venv/bin/activate
echo "Python: $(which python)"

python $SCRIPT --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --pair $PAIR
python $SCRIPT --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --pair $PAIR --do-raw 1