#!/usr/bin/env bash
set -e

python -m p2p_peft_fl.cli \
  --model bert \
  --dataset sst2 \
  --peft lora \
  --nodes 3 \
  --topology ring \
  --protocol gossip \
  --rounds 3 \
  --local_steps 30 \
  --max_train_samples 2000 \
  --max_eval_samples 1000
