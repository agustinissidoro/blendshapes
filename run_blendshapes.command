#!/bin/zsh
set -e

# Ensure conda is available in this non-interactive shell
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Conda not found. Update this script with your conda.sh path."
  exit 1
fi

conda activate blendshapes
cd /Users/agustinissidoro/blendshapes
python main.py

# Keep window open if launched by double-click
read -n 1 -s -r -p "Press any key to close..."
