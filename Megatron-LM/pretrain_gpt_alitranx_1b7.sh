#!/bin/bash

#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

# Using sequence parallelism requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# debug
# WORLD_SIZE=1
# RANK=0
# MASTER_ADDR=localhost
# MASTER_PORT=9527

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"


# GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * data_parallel_size * grad_acc_steps
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=32
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1

# Rampup-batch-size
# START_BATCH_SIZE=MICRO_BATCH_SIZE * ${WORLD_SIZE}
# BATCH_SIZE_INCREMENT=${WORLD_SIZE} * 2
# NUM_RAMPUP_SAMPLES=300000

TRAIN_ITERS=610000
LR_DECAY_ITERS=608000
WARMUP_IERS=2000

TRAINING_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-warmup-iters $WARMUP_IERS \
    --clip-grad 1.0 \
    --weight-decay 0.0001 \
    --layernorm-epsilon 1e-5 \
    "

GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --init-method-std 0.006 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    --fp32-residual-connection \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --sequence-parallel \
    --recompute-activations
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 40 \
    --split 99.9,0.1,0 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-params-norm \
    --log-num-zeros-in-grad \
    "

python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    ./pretrain_gpt.py \
    --data-path "./bin/data_bin_text_document" \
    --data-impl "mmap" \
    --num-workers 1 \
    --tokenizer-type "SentencePieceTokenizer" \
    --vocab-file "./vocab.unigram.model" \
    --save ./save_model/run_polylm_13b_micro_8_${DATETIME} \
    --tensorboard-dir ./tensorboard/run_polylm_13b_micro_8_${DATETIME} \
    ${GPT_ARGS} \
    ${OUTPUT_ARGS} \
    ${TRAINING_ARGS} | tee -a ./logs/run_polylm_13b_micro_8_${DATETIME}_rank_${RANK}

