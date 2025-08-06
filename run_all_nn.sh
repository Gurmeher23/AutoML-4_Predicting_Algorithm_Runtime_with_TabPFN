#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

SCENARIOS=(
  clasp_factoring
  saps-CVVAR
  lpg-zeno
  spear_qcp
  spear_swgcp
  yalsat_qcp
  yalsat_swgcp
)
MODELS=(lognormal_nn.floc invgauss_nn.floc expon_nn.floc)

for scen in "${SCENARIOS[@]}"; do
  for model in "${MODELS[@]}";  do
    for fold in {0..9};         do
      echo "▶ $scen  $model  fold $fold"
      python scripts/eval_model.py \
             --model    "$model" \
             --scenario "$scen" \
             --fold     "$fold" \
             --save     ../DistNet_nn_pkl
    done
  done
done
