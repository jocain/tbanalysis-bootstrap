export RESULTS_DIR="results_fiducial"
INPUT_DIRS=(
    # "$RESULTS_DIR/Test-bootstrap_147124_147487_target-trigger-2_OPTSEL/step3/"
    # "$RESULTS_DIR/Test-bootstrap_147488_147565_target-trigger-2_OPTSEL/step3/"
    # "$RESULTS_DIR/Test-bootstrap_147566_147641_target-trigger-2_OPTSEL/step3/"
    # "$RESULTS_DIR/Test-bootstrap_147644_147885_target-trigger-2_OPTSEL/step3/"
    # "$RESULTS_DIR/Test-bootstrap_147886_147966_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_147967_148053_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_148847_148926_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_148927_149131_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_149132_149203_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_149204_149279_target-trigger-2_OPTSEL/step3/"
    "$RESULTS_DIR/Test-bootstrap_149280_149350_target-trigger-2_OPTSEL/step3/"
    # "${RESULTS_DIR}/Test-bootstrap_149436_149584_target-trigger-2_OPTSEL/step3/" #Bad
    # "$RESULTS_DIR/Test-bootstrap_149585_149665_target-trigger-2_OPTSEL/step3/"
)

python do_bootstrap_multiple_step4.py --input-dirs "${INPUT_DIRS[@]}" --output-dir $RESULTS_DIR/step4/
