#!/bin/bash
INPUT_DIR=$1
OUTPUT_DIR=$2
TRIGGER=$3
DOGLOBAL=$4

echo "Setting up environment..."
source $WORK_DIR/venv/bin/activate
echo "Python: $(which python)"
echo "python3 $SCRIPT --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --layer-i 0 --layer-j 1 --layer-k 2 --trigger $TRIGGER --use-best 1 $DOGLOBAL"
python3 $SCRIPT --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --layer-i 0 --layer-j 1 --layer-k 2 --trigger $TRIGGER --standardize-twc 5 --use-best 1 $DOGLOBAL
