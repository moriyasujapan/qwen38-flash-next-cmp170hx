#!/usr/bin/env bash
set -euo pipefail

echo '== GPUs =='
nvidia-smi --query-gpu=index,name,uuid,memory.total,power.limit --format=csv

echo
echo '== Power Brake =='
nvidia-smi -q | grep -A1 'HW Power Brake Slowdown' || true

echo
echo '== PCIe link =='
for device in /sys/bus/pci/devices/*; do
  [[ -r "$device/vendor" && $(<"$device/vendor") == 0x10de ]] || continue
  [[ -r "$device/current_link_speed" ]] || continue
  printf '%s: speed=%s width=%s\n' \
    "$(basename "$device")" \
    "$(<"$device/current_link_speed")" \
    "$(<"$device/current_link_width")"
done

echo
echo '== Memory =='
free -h

available_gib=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if (( available_gib < 52 )); then
  echo "WARNING: MemAvailable=${available_gib} GiB; 52 GiB or more is recommended." >&2
fi

echo
echo '== Docker GPU access =='
docker info >/dev/null
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L
