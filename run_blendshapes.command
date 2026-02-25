#!/bin/zsh
set -euo pipefail

# Always run from this script's directory so config/model relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure libraries that use cache/config paths can write on any machine.
export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$MPLCONFIGDIR" "${XDG_CACHE_HOME}/fontconfig"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/anaconda3/bin/conda"
else
  echo "Conda was not found. Install Conda and create the 'blendshapes' environment first."
  exit 1
fi

echo "Launching blendshapes from: $SCRIPT_DIR"
set +e
"$CONDA_BIN" run --no-capture-output -n blendshapes python main.py
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo ""
  echo "run_blendshapes.command failed with exit code ${EXIT_CODE}."
  echo "If launched by double-click, this window will stay open so you can read the error."
  read -r "REPLY?Press Enter to close..."
  exit "${EXIT_CODE}"
fi
