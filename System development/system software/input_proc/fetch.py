from .fetcher import *
from .receiver import *
from common import *
from time import time
import os
import psutil


def fetch(core, fetch_input_image,fetch_input_receive,fetch_input_image_lock,fetch_input_receive_lock,fpga_input,fpga_input_lock, label_set, real_limit_set, start_inference_time, stop_inference_time, pipeline_ready, pipeline_ready_lock):
    os.sched_setaffinity(0, {core})
    p = psutil.Process(os.getpid())
    p.nice(-20)

    if ROLE == MASTER:
        fetch_image(fetch_input_image,fetch_input_image_lock,fpga_input,fpga_input_lock, label_set, real_limit_set, start_inference_time, stop_inference_time, pipeline_ready, pipeline_ready_lock)
    elif ROLE == DEPENDENT:
        receive_input(fetch_input_receive,fetch_input_receive_lock,fpga_input,fpga_input_lock, pipeline_ready, pipeline_ready_lock)

