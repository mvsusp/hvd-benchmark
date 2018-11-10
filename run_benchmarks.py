import time

import click
from sagemaker.estimator import Estimator

from build_sagemaker_container import build, push


@click.command()
@click.option('--instance_count', type=int, required=True)
@click.option('--role', required=True)
@click.option('--base-image')
@click.option('--tag')
@click.option('--aws-account')
@click.option('--region')
@click.option('--subnet')
@click.option('--security_group')
@click.option('--build-image/--no-build', default=True)
def run_benchmark(instance_count,
                  subnet,
                  security_group,
                  aws_account,
                  base_image,
                  region='us-west-2',
                  role="SageMakerRole",
                  tag='tensorflow-hvd:latest',
                  build_image=False,
                  wait=True):

    if build_image:
        build(
            base_image=base_image,
            entrypoint='launcher.sh',
            source_dir='benchmarks',
            tag=tag)

    ecr_image_name = push(tag)

    output_path = 's3://sagemaker-{}-{}/hvd-1-single/{}node-{}'.format(
        region, aws_account, instance_count, time.time_ns())

    estimator = Estimator(
        ecr_image_name,
        role=role,
        base_job_name='hvd-bench',
        hyperparameters={},
        train_instance_count=instance_count,
        train_instance_type='ml.p3.16xlarge',
        output_path=output_path,
        subnets=[subnet],
        security_group_ids=[security_group])

    estimator.fit(
        's3://sagemaker-sample-data-%s/spark/mnist/train/' % region, wait=wait)


if __name__ == '__main__':
    run_benchmark()
