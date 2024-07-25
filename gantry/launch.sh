#!/usr/bin/env bash
set -ex

NUM_NODES=1

# ABLATION_GROUPS=("no_wiki" "baseline" "no_code" "no_math" "no_flan")
# ABLATION_GROUPS=("no_code" "no_math" "no_flan")
ABLATION_GROUPS=("no_flan" "no_math")
# "olmo-156m" "olmo-11m" "olmo-86m" olmo-508m
#TODO: 11m
MODEL_SIZES=("olmo-86m")
YMD="2024-07-21"

# ABLATION_GROUPS=("no_math")
# MODEL_SIZES=("olmo-86m")

# Ensure train.sh is executable
chmod +x gantry/train.sh

for MODEL in "${MODEL_SIZES[@]}"; do
    for GROUP in "${ABLATION_GROUPS[@]}"; do
        RUN_NAME="${GROUP}_${MODEL}"
        CONFIG_PATH="configs/tain/${YMD}/${GROUP}_${MODEL}.yaml"
        gantry run \
          --workspace ai2/cheap_decisions \
          --task-name data-decisions \
          --description "${GROUP}_${MODEL}" \
          --priority high \
          --preemptible \
          --beaker-image tain/olmo-tai \
          --cluster ai2/jupiter-cirrascale-2 \
          --gpus 8 \
          --replicas "${NUM_NODES}" \
          --leader-selection \
          --host-networking \
          --budget ai2/oe-training \
          --no-nfs \
          --propagate-failure \
          --env LOG_FILTER_TYPE=local_rank0_only \
          --env OMP_NUM_THREADS=8 \
          --env OLMO_TASK=model \
          --env-secret WANDB_API_KEY=WANDB_API_KEY_TAI \
          --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
          --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
          --shared-memory 10GiB \
          --venv base \
          --mount /data:/data \
          --yes \
          --timeout=-1 \
          --allow-dirty \
          -- /bin/bash -c "gantry/train.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK ${CONFIG_PATH}" &
    done
done
