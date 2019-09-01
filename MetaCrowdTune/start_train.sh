#!/usr/bin/env bash

DATASET="WorldExpoMeta"
TRAIN_PREFIX="python train.py --dataset $DATASET"

DATA_PATH="/home/bhuniaa/Project/WE10/train/"
NUM_TASKS=1
NUM_INSTANCES=1
META_BATCHSIZE=32
BASE_BATCHSIZE=1
META_LR=0.001
BASE_LR=0.001
EPOCHS=15000
ARCHITECTURE="csr"
BASE_UPDATES=5
EXPERIMENT=101
LOG_DIR="./logs/maml_1shot_1scene_32batch_1e-3lr_5base_updates_exp_101.log"

CMD="$TRAIN_PREFIX --data_path $DATA_PATH --num_tasks $NUM_TASKS --num_instances $NUM_INSTANCES --meta_batch $META_BATCHSIZE --base_batch $BASE_BATCHSIZE --meta_lr $META_LR --base_lr $BASE_LR --epochs $EPOCHS --architecture $ARCHITECTURE --base_updates $BASE_UPDATES --experiment $EXPERIMENT --log $LOG_DIR"

echo $CMD; eval $CMD
