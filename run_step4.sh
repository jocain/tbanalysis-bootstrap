INPUT_DIRS=(
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147124_147487/step3
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147488_147565/step3
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147566_147641/step3
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147644_147885/step3
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147967_148053/step3
  /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/ResultsTWCUpdated_v2_147886_147966/step3
)

python do_bootstrap_multiple_step4.py --input-dirs "${INPUT_DIRS[@]}" --output-dir /eos/user/f/fernance/ETL/Bootstrap/July2026/Results/step4-Updated
