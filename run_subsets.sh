#!/usr/bin/env bash
set -euo pipefail

# where to save your .pkl outputs
OUTROOT=./DistNet_nn_pkl_subsets

# ensure the parent dir exists
mkdir -p "$OUTROOT"

# which scenarios to sweep
SCENARIOS=(
  clasp_factoring
  saps-CVVAR
  lpg-zeno
  spear_qcp
  spear_swgcp
  yalsat_qcp
  yalsat_swgcp
)

# how many cross‐val folds
FOLDS=({0..9})

# the different # of samples per instance to try
SUBSET_SIZES=(1 2 5 10 20 50 100)

# loop
for scen in "${SCENARIOS[@]}"; do
  for n in "${SUBSET_SIZES[@]}"; do
    # make a subdirectory per N
    OUTDIR="$OUTROOT/subset${n}"
    mkdir -p "$OUTDIR"
    for fold in "${FOLDS[@]}"; do
      echo "▶ Scenario=$scen  fold=$fold  samples=$n"
      python scripts/eval_model.py \
        --model            lognormal_nn.floc \
        --scenario         "$scen" \
        --fold             "$fold" \
        --num_train_samples "$n" \
        --save             "$OUTDIR"
    done
  done
done