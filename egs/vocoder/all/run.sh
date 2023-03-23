#!/bin/bash

ln -sf ../../../utils
chmod +x utils/*

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
verbose=1      # verbosity level (lower is less info)
n_gpus=4       # number of gpus in training

master_port=29500 

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan.v1.yaml

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
pretrain="" # checkpoint path to load pretrained parameters
            # (e.g. ../../jsut/<path>/<to>/checkpoint-400000steps.pkl)


# dataset
train_scp="data/train.scp"
dev_scp="data/valid.scp"

# parse argv
. utils/parse_options.sh || exit 1;

# force-modified number of gpus
if [ ! -z ${CUDA_VISIBLE_DEVICES+x} ]; then
    arr=(${CUDA_VISIBLE_DEVICES//,/ })
    n_gpus=${#arr[*]}
fi

set -euo pipefail

if [ -z "${tag}" ]; then
    expdir="exp/$(basename "${conf}" .yaml)"
    if [ -n "${pretrain}" ]; then
        pretrain_tag=$(basename "$(dirname "${pretrain}")")
        expdir+="_${pretrain_tag}"
    fi
else
    expdir="exp/${tag}"
fi


echo "Start: Network training"
[ ! -e "${expdir}" ] && mkdir -p "${expdir}"
if [ "${n_gpus}" -gt 1 ]; then
    train="python3 -m dialtts.vocoder.distributed.launch --nproc_per_node ${n_gpus} --master_port ${master_port} -c dialtts-vocoder-train"
else
    train="dialtts-vocoder-train"
fi
echo "Training start. See the progress via ${expdir}/train.log."
${cuda_cmd} "${expdir}/train.log" \
    ${train} \
        --config "${conf}" \
        --train-scp "${train_scp}" \
        --dev-scp "${dev_scp}" \
        --outdir "${expdir}" \
        --resume "${resume}" \
        --pretrain "${pretrain}" \
        --verbose "${verbose}"
echo "Successfully finished training."

echo "Finished."
