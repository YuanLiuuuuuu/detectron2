# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import warnings

import cv2
import torch.multiprocessing as mp
import torch
import subprocess


def setup_multi_processes(args):
    args = vars(args)
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = args.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = args.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and 2 > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and 2 > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def init_distributed_mode():
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29400')
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['LOCAL_SIZE'] = str(num_gpus)

    local_rank = proc_id % num_gpus

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=ntasks, rank=proc_id)
    torch.distributed.barrier(device_ids=[local_rank])
    return local_rank
