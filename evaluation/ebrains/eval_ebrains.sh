#!/bin/bash
if [ "${CONDA_DEFAULT_ENV:-}" != "py310" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py310
fi
python linear_eval_prob_fine.py &&
python linear_eval_prob_coarse.py
