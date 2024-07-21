#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

CONFIG_PATH=$1
shift

source scripts/beaker/warm_hf_cache.sh

# Use all interfaces starting with `ib`. This selects the IB cards and avoids
# interfaces with names like bond0 and enp0, which are usually ethernet devices.
# Ethernet networks are not robust/fast enough for most distributed training workloads.
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME=ib

# Don't use the IB bond (which uses the attached ethernet cards) for the same reason.
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca
export NCCL_IB_HCA="^=mlx5_bond_0"

# Enable additional NCCL output for debugging.
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug
export NCCL_DEBUG=INFO

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12347 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py $CONFIG_PATH \
  --save-overwrite \
  --save_overwrite
