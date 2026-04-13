#!/bin/bash
# Pull latest changes from the blendshapes repo
# Double-click this file on macOS to run it

REPO_URL="https://github.com/agustinissidoro/blendshapes.git"
TARGET_DIR="$HOME/blendshapes"

echo "=== Pulling blendshapes ==="

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Repo found at $TARGET_DIR — pulling latest..."
    cd "$TARGET_DIR" && git pull
else
    echo "Repo not found — cloning to $TARGET_DIR..."
    git clone "$REPO_URL" "$TARGET_DIR"
fi

echo ""
echo "Done! Press any key to close."
read -n 1 -s
