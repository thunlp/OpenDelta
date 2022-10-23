#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="./"
VERSION="bert-large-cased"
DATASET="BoolQ"  # You can try other dataset listed in https://github.com/OpenBMB/ModelCenter/tree/main/examples/bert

OPTS=""
OPTS+=" --model-config ${VERSION}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 64"
OPTS+=" --lr 0.001" # You can use different learning rate to find optimal performance 
OPTS+=" --max-encoder-length 512"
OPTS+=" --train-iters 1400"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 128"
OPTS+=" --delta_type low_rank_adapter" # You can use different delta type, listed in https://opendelta.readthedocs.io/en/latest/notes/acceleration.html#BMTrain

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}2_with_bmtrain.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/tmp/logs/bmt_bert_boolq_finetune-${VERSION}-${DATASET}.log

