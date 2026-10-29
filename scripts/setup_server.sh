#!/usr/bin/env bash
# Prepare the VPS once, as root. Safe on a server that already runs other stacks:
# it only adds swap (if there is none) and checks Docker. It does not touch nginx or the firewall.
set -euo pipefail

# Without swap, a memory spike makes the kernel OOM killer pick a process, possibly another stack's database.
if [ -z "$(swapon --show)" ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Use swap only as a safety net, not as regular memory.
  sysctl -w vm.swappiness=10
  echo 'vm.swappiness=10' > /etc/sysctl.d/99-swappiness.conf
fi
swapon --show

docker --version
docker compose version
