#!/usr/bin/env bash
# run_all_lognormal.sh  – produce 7 scenarios × 10 folds = 70 *.pkl files

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
MODELS=(lognormal_distfit.floc)   # baseline model
SCENARIOS=(
  clasp_factoring
  saps-CVVAR
  lpg-zeno
  spear_qcp
  spear_swgcp
  yalsat_qcp
  yalsat_swgcp
)
for scen in "${SCENARIOS[@]}"; do
  for fold in {0..9}; do
    echo "▶ $scen  fold $fold"
    python scripts/eval_model.py \
           --model lognormal_distfit.floc \
           --scenario "$scen" \
           --fold "$fold" \
           --save ../
  done
done