#!/usr/bin/env bash

python -m venv new-env
source new-env/bin/activate

pip install pip -U
pip install -r requirements.txt

# build the image
python run_benchmarks.py   --instance_count 4 --role SageMakerRole --tag tensorflow-hvd:latest --base-image mvsusp/hvd-benchmark \
  --aws_account 369233609183 --region us-west-2 --subnet subnet-125fb674 --security_group sg-ce5dd1b4

# dont build
python run_benchmarks.py   --instance_count 4 --role SageMakerRole --tag tensorflow-hvd:latest --base-image mvsusp/hvd-benchmark --aws-account 369233609183 --region us-west-2 --subnet subnet-125fb674 --security_group sg-ce5dd1b4 --no-build