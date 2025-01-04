


from pynq import PL
PL.reset()
import multiprocessing
import threading
from communication.com_fetch_input_image import *
from communication.com_fetch_input_recv import *
from communication.com_post_forward import *
from communication.com_post_recv import *
import socket
from common import *
from input_proc.fetch import fetch
from output_proc.post import post
from inference_proc.inference import inference
import time
import os
import psutil

fetch_core = 1
inference_core = 2
post_core = 3
master_recv_core = 0

os.sched_setaffinity(0, {post_core})
p = psutil.Process(os.getpid())
p.nice(-20)

# guard
num_data_buffer_fpga = multiprocessing.Value('i', 0)
num_out_buffer_fpga = multiprocessing.Value('i', 0)


# shared memory
fpga_input = multiprocessing.Manager().list()
fpga_output = multiprocessing.Manager().list()
label_set = multiprocessing.Manager().list()
real_limit_set = multiprocessing.Manager().list()
pipeline_ready = multiprocessing.Manager().list()

fpga_input_lock = multiprocessing.Lock()
fpga_output_lock = multiprocessing.Lock()
pipeline_ready_lock = multiprocessing.Lock()


result_list = multiprocessing.Manager().list()
start_inference_time = multiprocessing.Manager().list()
stop_inference_time = multiprocessing.Manager().list()

# communication input output
fetch_input_image = multiprocessing.Manager().list()    # MASTER
fetch_input_receive = multiprocessing.Manager().list()  # DEPENDENT
post_forward =  multiprocessing.Manager().list()        # each device
post_output_receive =  multiprocessing.Manager().list() # master

fetch_input_image_lock = multiprocessing.Lock()
fetch_input_receive_lock = multiprocessing.Lock()
post_forward_lock = multiprocessing.Lock()
post_output_receive_lock = multiprocessing.Lock()

# process
process1 = multiprocessing.Process(target=fetch, args=(fetch_core , fetch_input_image,fetch_input_receive,fetch_input_image_lock,fetch_input_receive_lock,fpga_input,fpga_input_lock, label_set, real_limit_set, start_inference_time, stop_inference_time, pipeline_ready, pipeline_ready_lock))
process2 = multiprocessing.Process(target=inference, args=(inference_core , fpga_input, fpga_input_lock, fpga_output, fpga_output_lock, start_inference_time,pipeline_ready, pipeline_ready_lock))
#process3 = multiprocessing.Process(target=post, args=(fpga_output, fpga_output_lock, post_forward, post_output_receive, post_forward_lock, post_output_receive_lock)) 
# TODO: set core for each process
if ROLE == MASTER:
    process4 = multiprocessing.Process(target=com_post_recv, args=(master_recv_core, fpga_output, fpga_output_lock, post_output_receive, post_output_receive_lock, label_set, real_limit_set, start_inference_time, stop_inference_time,pipeline_ready, pipeline_ready_lock))

# handling communication
"""
prev_socket = []  # Connection to previous device. initialized within fetch process
next_socket = None  # Connection to next device
host_socket = None


if DEVICE_NUMBER > 1:
    prev_socket.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    next_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("preparing the host socket")
    # initialize localhost port
    host_socket.bind((host_ip, host_port))

    print("listening ...")
    host_socket.listen(1)
    print("connecting to the next host ...")
    while True:
        try:
            next_socket.connect((next_host_ip, next_port))
            break
        except ConnectionRefusedError:
            time.sleep(5)
    print("connected to the next host")
    prev_socket[0], prev_address = host_socket.accept() 
    print("connected to the pervious host")

"""

#thread1 = threading.Thread(target=com_fetch_input_image, args=(fetch_input_image, fetch_input_image_lock, prev_socket))   # MASTER - directly embedded into fetch process since the input image only simulation

# post is apart of this process
#thread3 = threading.Thread(target=com_post_forward, args=(post_forward, post_forward_lock, fpga_output, fpga_output_lock))


# Start the process
process1.start()
process2.start()
# Start the threads
#thread1.start()
#thread3.start()
if ROLE == MASTER:
    process4.start()

# post is apart of this process
com_post_forward(post_forward, post_forward_lock, fpga_output, fpga_output_lock, pipeline_ready, pipeline_ready_lock,stop_inference_time, start_inference_time)

# Wait for all threads to complete
#thread1.join()
#thread3.join()
if ROLE == MASTER:
    process4.join()

process1.join()
process2.join()