#!/bin/bash
set -e

echo "Setting up Vault authentication..."

# Install dependencies if missing
if ! which curl >/dev/null 2>&1 || ! which unzip >/dev/null 2>&1; then
  echo "Installing required dependencies..."
  if which micromamba >/dev/null 2>&1; then
    # Use micromamba if available (docker runners)
    micromamba install -y -c conda-forge curl unzip
  elif which apk >/dev/null 2>&1; then
    # Alpine
    apk add --no-cache curl unzip
  elif which apt-get >/dev/null 2>&1; then
    # Debian/Ubuntu
    DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl unzip
  elif which yum >/dev/null 2>&1; then
    # RHEL/CentOS
    yum install -y curl unzip
  elif which dnf >/dev/null 2>&1; then
    # Fedora
    dnf install -y curl unzip
  else
    echo "Warning: Could not detect package manager. Please ensure curl and unzip are available."
  fi
fi

# Download vault agent if not available

echo "Downloading Vault agent..."
rm -f vault_agent.zip vault
curl https://urm.nvidia.com/artifactory/sw-kaizen-data-generic-local/com/nvidia/vault/vault-agent/2.4.4/nvault_agent_v2.4.4_linux_amd64.zip -L -o vault_agent.zip
unzip vault_agent.zip
chmod +x vault

# Verify vault binary
./vault -version

# Run vault agent to authenticate and render secrets
echo "Authenticating with Vault and retrieving secrets..."
./vault agent -config=./vault-ci/vault-agent.config -exit-after-auth

# Verify secrets file was created
if [ ! -f "./ci_secrets.env" ]; then
  echo "Error: Vault secrets file not created"
  exit 1
fi

# Load secrets into environment
source ./ci_secrets.env
echo "Vault secrets retrieved and loaded successfully"

# Clean up files now that they're environment variables
rm -f ./ci_secrets.env
echo "Secrets files cleaned up"
