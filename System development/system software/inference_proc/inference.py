
from common import *
from util.driver_base import FINNExampleOverlay
import time
import os
import psutil


def inference(core, fpga_input, fpga_input_lock, fpga_output, fpga_output_lock, start_inference_time,pipeline_ready, pipeline_ready_lock):
    os.sched_setaffinity(0, {core})
    p = psutil.Process(os.getpid())
    p.nice(-20)

    print("preparing the accelerator")
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir, device=device
    )
    print("successfully load the accelerator")
    with pipeline_ready_lock:
        pipeline_ready.append(1)
    start_inference_time.append(0) # test
    for i in range(len(start_inference_time)):
        start_inference_time.pop(i)
    while len(pipeline_ready) != number_of_process: #make sure all the process are ready before staring the pipeline or inference
        pass
    while True:
        if len(fpga_input) > 0:                                         # if there is an input 
            with fpga_input_lock:
                accel.copy_input_data_to_device(fpga_input[0])          # put it into buffer
                fpga_input.pop(0)
            #print(f"start_inference_time:    {start_inference_time[len(start_inference_time)-1]}")
            
            start_inference_time.append(time.time_ns()) 
            accel.execute_on_buffers()   
            #start_0 = time.time_ns()
            output_pack = np.empty_like(accel.obuf_packed_device[0])    # get pack output
            accel.copy_output_data_from_device(output_pack)
            #stop = time.time_ns()
            obuf_folded = accel.unpack_output(output_pack)              # get unpack output (folded)
            #stop_2 = time.time_ns()
            #print(f"copy time:      {((stop - start_0)*10**(-9))/16} s")
            #print(f"expand time:    {((stop_2 - stop)*10**(-9))/16} s")
            #finish_accel_time = time.time_ns()
            #print(f"finish_accel_time:              {finish_accel_time}")
            #print(f"from start to forward:  {finish_accel_time - start_inference_time[len(start_inference_time)-1]}")
            if len(fpga_output) <= 0:                                   # if fpga output queue is empty
                # either forward unpackdata or direct inteprate on pack data if only one device
                if DEVICE_NUMBER > 1:
                    with fpga_output_lock:
                        fpga_output.append(obuf_folded)
                elif DEVICE_NUMBER == 1:
                    with fpga_output_lock:
                        fpga_output.append(output_pack)
                else:
                    print("number of device is not valid. stopping inference")
                    break







       

