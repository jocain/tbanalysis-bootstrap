#!/bin/bash
RUN_START=$1
RUN_STOP=$2
TRIGGER=$3
EFF_LAYER=$4
OUTDIR=$5

echo "Setting up environment..."
source $WORK_DIR/venv/bin/activate
echo "Python: $(which python)"

if [ -n "$OUTDIR" ]; then
    python $SCRIPT $RUN_START $RUN_STOP $TRIGGER $EFF_LAYER --output-dir $OUTDIR
else
    python $SCRIPT $RUN_START $RUN_STOP $TRIGGER $EFF_LAYER
fi
