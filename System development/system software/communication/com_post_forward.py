from common import *
import time
import socket

def com_post_forward(post_forward, post_forward_lock, fpga_output, fpga_output_lock, pipeline_ready, pipeline_ready_lock, stop_inference_time, start_inference_time):
    if DEVICE_NUMBER > 1:
        next_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("connecting to the next host ...")
        while True:
            try:
                next_socket.connect((next_host_ip, next_port))
                break
            except ConnectionRefusedError:
                time.sleep(5)
                print("connecting to the next host ...")
        print("connected to the next host")
        
        with pipeline_ready_lock:
            pipeline_ready.append(1)

        while True:
            if len(fpga_output) > 0:
                uint8_data = None
                with fpga_output_lock:
                    uint8_data = fpga_output[0].astype(np.uint8)    # transform into uint8
                    fpga_output.pop(0)
                #send_time = time.time_ns()
                #print(f"ready send time:              {send_time}")
                #print(f"from start to ready forward:  {send_time - start_inference_time[len(stop_inference_time)]}")
                data_to_send = uint8_data.tobytes()             # transform into bytes
                next_socket.sendall(data_to_send)               # foward to next socket
                #send_time = time.time_ns()
                #print(f"send time:              {send_time}")
                #print(f"from start to forward:  {send_time - start_inference_time[len(stop_inference_time)]}")
    elif DEVICE_NUMBER == 1:
         print("stopping com_post_forward. the service is not required")
        
    else:
         print("stopping com_post_forward. invalid number of device")
