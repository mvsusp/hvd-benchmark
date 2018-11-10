#!/usr/bin/env python
#
# # Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os

import shlex
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time

from contextlib import contextmanager
import click

import sagemaker_containers.beta.framework as framework
from retrying import retry

logging.basicConfig(
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"

MODEL_FILE_NAME = "model.npz"


@click.command()
@click.option('--program', required=True)
@click.argument('args', nargs=-1)
def cli(program, args):
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)

    logger.setLevel(env.log_level)
    train(env, hyperparameters, program, args)


class TimeoutError(Exception):
    pass


def train(env, hyperparameters, program, args):
    current_host = env.current_host
    hosts = list(env.hosts)
    _change_hostname(current_host)

    # Allow processes launched by mpirun to assume SageMaker IAM role
    _expose_aws_credential_chain_to_mpi_processes()

    _start_ssh_daemon()

    if current_host == _get_master_host_name(hosts):
        _wait_for_worker_nodes_to_start_sshd(hosts)

        _run_mpi_on_all_nodes(env, hyperparameters, program, args)
    else:
        _wait_for_training_to_finish(env)


def _expose_aws_credential_chain_to_mpi_processes():
    """
    At runtime, SageMaker sets an envvar (unique per node) in each container that adds the IAM role to the AWS
    credential chain. When mpirun launches processes on a remote host, those processes do not see that envvar
    and AWS SDK calls fail due to lack of credentials. So we add the envvar to the beginning of .bashrc to expose it to
    the mpirun processes
    """
    with open('/root/.bashrc.new', 'w+') as new_bashrc:
        new_bashrc.write(
            'export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI={}\n'.format(
                os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")))
    subprocess.check_output(
        "cat /root/.bashrc >> /root/.bashrc.new", shell=True)
    subprocess.check_output("mv /root/.bashrc.new /root/.bashrc", shell=True)
    subprocess.check_output("cat /root/.bashrc", shell=True)


def _run_training(env):
    logger.info('Invoking user training script.')

    framework.modules.run_module(env.module_dir, env.to_cmd_args(),
                                 env.to_env_vars(), env.module_name)


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.

    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh {}".format(current_host))


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _run_mpi_on_all_nodes(env, hyperparameters, program, args):
    mpi_command = _get_mpi_command(env, hyperparameters, program, args)

    framework.logging.log_script_invocation(mpi_command, env.to_env_vars(),
                                            logger)

    outfile = '/opt/ml/model/log.log'
    with open(outfile, 'w') as o:
        o.write(mpi_command)

        try:
            out = subprocess.check_output(mpi_command, shell=True)
            o.write("\n")

            result = out.decode("utf-8")

            lines = result.split("\n")

            def keep_good_messages(item):
                return not "Read -1" in item

            result = "\n".join(
                [item for item in lines if keep_good_messages(item)])

            o.write(result)
            for l in result.split("\n"):
                print(l)
            print(result)
            exitcode = 0
            e = "----NO ERROR----"
            print(str(e))
            o.write("\n")
            o.write(str(e))
        except Exception as e:
            print("exception occured")
            exitcode = 1
            print(str(e))
            o.write("\n")
            o.write(str(e))
    with open(outfile, 'r') as o:

        print(o.readlines())

    sys.exit(exitcode)


def _get_mpi_command(env, hyperparameters, program, args):
    """Constructs a command to run distributed training with MPI using mpirun.

    Runs /mpi_script.sh on all hosts listed in the training environment. How many
    processes in total is determined by the 'sagemaker_num_processes' hyperparameter, or one
    per GPU, or one per CPU, as applicable. The 'sagemaker_process_slots_per_host'
    hyperparameter can be used to override how many processes can be placed on each host.

    Additional MPI options can be passed (and override other MPI options) using the
    'sagemaker_additional_mpi_options' hyperparameter.

    This command passes many options to the mpirun command:

    * --host [host:slots]: A comma-delimited list of hosts and the number of process
        slots on each host.
    * -mca btl_tcp_if_include [env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for byte transfer layer communication.
    * -mca oob_tcp_if_include [env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for out-of-band communication.
    * -mca btl ^openib: Don't look for openib components (this just avoids a warning)
    * -x PATH: pass $PATH from the current environment to the execution environments on remote hosts
    * -x LD_LIBRARY_PATH: pass $LD_LIBRARY_PATH from the current environment to the execution
        environments on remote hosts
    * -x LD_PRELOAD=[changehostname library]: Load the changehostname library to return
        correct values from gethostname system calls.
    * -mca orte_abort_on_non_zero_status 1: Return a non-zero exit code if any process exits
        with a non-zero exit code.
    * -x NCCL_DEBUG=INFO: Enable info level logging for NCCL.
    * -x NCCL_SOCKET_IFNAME=[env.network_interface_name]: Tell NCCL to use the given
        network interface name for socket communication.
    * -np [num_processes]: total number of processes to run across all nodes.

    Args:
        env: training environment object containing environment variables,
                              training arguments and hyperparameters.

    Returns:
        str: The mpirun command to run.
    """
    is_gpu = env.num_gpus if env.num_gpus > 0 else 1

    process_slots_per_host = int(
        hyperparameters.get('sagemaker_process_slots_per_host', is_gpu))

    num_hosts = len(env.hosts)
    num_processes = process_slots_per_host * num_hosts
    num_processes = int(
        hyperparameters.get('sagemaker_num_processes', num_processes))

    # By default, use one process per GPU, or one process per node (if training with CPU).
    host_list = env.hosts if process_slots_per_host == 1 else \
        [host + ':{}'.format(process_slots_per_host) for host in env.hosts]

    additional_mpi_options = str(
        hyperparameters.get('sagemaker_additional_mpi_options', ''))

    # credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']

    logger.info('network interface name: %s', env.network_interface_name)

    def build_host_arg(host_list, gpu_per_host):
        if len(host_list) == 1:
            return 'localhost:{}'.format(gpu_per_host)
        arg = ""
        for ind, host in enumerate(host_list):
            if ind != 0:
                arg += ","
            arg += '{}:{}'.format(host, gpu_per_host)
        return arg

    mpi_command = 'mpirun --allow-run-as-root --host {}'.format(
        build_host_arg(host_list, process_slots_per_host))
    " -mca btl_tcp_if_include eth0"
    " -mca oob_tcp_if_include eth0"
    " -x PATH"
    " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY)
    " -mca orte_abort_on_non_zero_status 1"
    " -x NCCL_DEBUG=INFO"
    " -x NCCL_SOCKET_IFNAME=eth0"
    " --mca plm_rsh_no_tree_spawn 1"
    " -bind-to none -map-by slot"
    " -mca pml ob1 -mca btl ^openib"
    " --display-map"
    " --tag-output"
    " -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO"
    " -x LD_LIBRARY_PATH -x PATH"
    " -np {} ".format(num_processes)
    " -x S3_REGION=us-west-2"
    " -x S3_ENDPOINT=us-west-2"
    " -x TF_CPP_MIN_LOG_LEVEL=0"
    " -x S3_USE_HTTPS=1"
    " --output-filename /opt/ml/model/hlog"

    mpi_command = "mpirun -np {} --host {} --allow-run-as-root --display-map --tag-output ".format(num_processes,
                                                                                                   ",".join(
                                                                                                       host_list)) + \
                  "-mca btl_tcp_if_include eth0 -mca oob_tcp_if_include eth0 -x NCCL_SOCKET_IFNAME=eth0 " + \
                  "--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib " + \
                  "-mca orte_abort_on_non_zero_status 1 -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO " + \
                  "-x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD={} --output-filename ".format(_CHANGE_HOSTNAME_LIBRARY) + \
                  "/opt/ml/model/hlog "

    #   + " --disable-dlopen " \

    # for v in credential_vars:
    #     if v in os.environ:
    #         mpi_command += " -x {}".format(v)

    # for name, value in env.to_env_vars().items():
    #     mpi_command += ' -x {}="{}"'.format(name, value)

    script = _create_mpi_script(env, program, args)

    mpi_command = mpi_command + " " + additional_mpi_options + " " + script

    return mpi_command


def _create_mpi_script(env, program, args):
    """Creates a MPI script with user provided information.

        For distributed training: the 'master node' runs mpirun with this script,
        '/mpi_script.sh'.

        This script creates a file '/mpi_is_running' that worker nodes use to
        determine whether training # (started by MPI from the master node) is still running.

        Processes on worker nodes use # /mpi_is_finished file to determine when to exit.

    Args:
        env (TrainingEnv): an instance of the training environment.
    """
    hyperparameters = framework.mapping.to_cmd_args(env.hyperparameters)

    python_cmd = [sys.executable, program]
    python_cmd.extend(hyperparameters)
    python_cmd.extend(args)

    content = textwrap.dedent("""#!/usr/bin/env bash
touch /mpi_is_running
%s 
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
""" % ' '.join(python_cmd))

    content = "sh -c 'touch /mpi_is_running && %s && EXIT_CODE=$? && touch /mpi_is_finished && exit ${EXIT_CODE}'" % ' '.join(
        python_cmd)

    return content


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _wait_for_training_to_finish(env):
    current_host = env.current_host

    logger.info("worker node %s is waiting for MPI to start training process",
                current_host)
    _wait_for_mpi_to_start_running()

    logger.info("MPI started training process on worker node %s", current_host)

    _wait_until_mpi_stops_running()
    logger.info("Training process started by MPI on worker node %s stopped",
                current_host)


def _wait_for_worker_nodes_to_start_sshd(hosts,
                                         interval=1,
                                         timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        logger.debug("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host %s", host)
        return True
    except socket.error:
        logger.debug("can't connect to host %s", host)
        return False


def _retry_if_false(result):
    return result is False


@retry(
    stop_max_delay=30 * 60 * 1000,
    wait_fixed=1000,
    retry_on_result=_retry_if_false)
def _wait_for_mpi_to_start_running():
    is_running = os.path.isfile(_MPI_IS_RUNNING)

    if is_running:
        print('MPI is running')
    else:
        print('waiting for nodes to connect')
    return os.path.isfile(_MPI_IS_RUNNING)


@retry(wait_fixed=5000, retry_on_result=_retry_if_false)
def _wait_until_mpi_stops_running():
    return os.path.isfile(_MPI_IS_FINISHED)


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    sm_openmpi()
