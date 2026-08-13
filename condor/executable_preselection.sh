#!/bin/bash
RUN_START=$1
RUN_STOP=$2
TRIGGER=$3
DOGLOBAL=$4

echo "Setting up environment..."
source $WORK_DIR/venv/bin/activate
echo "Python: $(which python)"

python $SCRIPT $RUN_START $RUN_STOP $TRIGGER $DOGLOBAL
