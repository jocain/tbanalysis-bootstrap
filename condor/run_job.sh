#!/bin/bash
ROW=$1
COL=$2

WORK_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
source $WORK_DIR/venv/bin/activate

python $WORK_DIR/do_bootstrap_multiple_fullstat.py \
    --row $ROW \
    --col $COL \
    --input-dir /eos/... \
    --output-dir /eos/.../output_${ROW}_${COL}