#!/usr/bin/env bash

pip install click retrying==1.3.3

cd sm
pip install .
cd -

sm-openmpi --program scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py -- \
            --num_batches=1000 \
            --model vgg16 \
            --batch_size 64 \
            --variable_update horovod \
            --horovod_device gpu \
            --use_fp16

