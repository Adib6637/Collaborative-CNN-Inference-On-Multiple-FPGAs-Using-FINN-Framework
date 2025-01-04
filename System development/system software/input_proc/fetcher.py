from util.dataset_transformer import dataset_transforms
from common import *
import time



def fetch_image(fetch_input_image,fetch_input_image_lock,fpga_input,fpga_input_lock, label_set, real_limit_set, start_inference_time, stop_inference_time, pipeline_ready, pipeline_ready_lock):
    print("preparing the input data")
    dataset, label_set_, real_limit_ = dataset_transforms(dataset_path, accel_ibuf_packed_device_shape)
    for l in label_set_:
        label_set.append(l)
    for rl in real_limit_:
        real_limit_set.append(rl)
    print("the input data are ready")
    b_n = len(dataset)
    with pipeline_ready_lock:
        pipeline_ready.append(1)
    while True:
        for i in range(b_n):
            while len(fpga_input) > 0:                              # if input buffer in fpga is empty
                pass
            with fpga_input_lock:
                fpga_input.append(dataset[i])                     # put the input into buffer
            #start_inference_time.append(time.time_ns())   
        print("all sample are used. stop fetching")

        while len(stop_inference_time) < len(dataset):
            time.sleep(1)
        
        print(f"start_inference_time:       {start_inference_time}")
        inference_time = 0
        total_inference_time = 0
        for i in range(len(stop_inference_time)):
            print(f"start_inference_time[i]:    {start_inference_time[i]}")
            print(f"stop_inference_time[i]:     {stop_inference_time[i]}")
            inference_time = (stop_inference_time[i]-start_inference_time[i])/batch_size
            print(f"inference_time:             {inference_time}")
            total_inference_time += (stop_inference_time[i]-start_inference_time[i])/batch_size
        average_inference_time = total_inference_time/len(dataset)
        print(f"average_inference_time:     {average_inference_time}")

        break



            

            
            
                                  


