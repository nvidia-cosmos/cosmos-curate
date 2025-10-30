#!/usr/bin/env bash
set -euo pipefail

# Cleanup NVCF function
if [[ "$HELM_DEBUG_KEEP_FAILED_DEPLOYMENTS" != "True" ]]; then
  if [ ! -f ~/.config/cosmos_curate/funcid.json ]; then
    echo "$HOME/.config/cosmos_curate/funcid.json not found, using working directory copy"
    if [ ! -f funcid_working.json ]; then
      echo "Error: funcid_working.json not found in working directory"
      exit 1
    fi
    mkdir -p ~/.config/cosmos_curate/
    cp funcid_working.json ~/.config/cosmos_curate/funcid.json
  fi
  cosmos-curate nvcf function delete-function
else
  echo "Intentionally leaving deployment behind for debugging. This must be manually cleaned up later."
fi
