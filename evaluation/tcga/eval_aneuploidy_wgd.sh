#!/bin/bash
if [ "${CONDA_DEFAULT_ENV:-}" != "py310" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py310
fi
python linear_eval_aneuploidy_wgd.py &&
python linear_reg_aneuploidy.py
