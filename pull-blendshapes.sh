#!/bin/bash
# Pull latest from blendshapes — always overwrites local with remote
# To make clickable: right-click → Open With → Terminal

REPO_URL="https://github.com/agustinissidoro/blendshapes.git"
TARGET_DIR="$HOME/blendshapes"

echo "=== Syncing blendshapes ==="

if [ -d "$TARGET_DIR/.git" ]; then
    cd "$TARGET_DIR"
    git fetch origin
    git reset --hard origin/main
    echo "Updated to latest."
else
    git clone "$REPO_URL" "$TARGET_DIR"
    echo "Cloned fresh."
fi

echo ""
echo "Done! Press any key to close."
read -n 1 -s
