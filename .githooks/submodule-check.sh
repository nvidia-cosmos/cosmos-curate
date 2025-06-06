#!/bin/bash

# This script is used in pre-commit hook to prevent accidental inclusion of
# cosmos-xenna in the MR

SUBMODULE_DIR="cosmos-xenna"

if git diff --cached --name-only | grep -q "^${SUBMODULE_DIR}"; then
  read -p " " answer < /dev/tty
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborting commit, Please remove ${SUBMODULE_DIR} from git staging area."
    exit 1
  fi
fi

exit 0
