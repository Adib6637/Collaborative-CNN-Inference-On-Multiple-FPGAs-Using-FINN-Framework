from common import *
import time
import socket
import os
import psutil

def com_post_recv(core, fpga_output, fpga_output_lock, post_output_receive, post_output_receive_lock, label_set, real_limit_set, start_inference_time, stop_inference_time, pipeline_ready, pipeline_ready_lock):  # fro now only for 1 bit model
    
    os.sched_setaffinity(0, {core})
    p = psutil.Process(os.getpid())
    p.nice(-20)
    
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    print("preparing the host socket")
    # initialize localhost port
    host_socket.bind((host_ip, host_port))

    print("listening ...")
    host_socket.listen(1)
    prev_socket, prev_address = host_socket.accept() 
    print("connected to the pervious host")

    total_correct = 0
    total_wrong = 0
    label_batch = 0
    # wait output from client to be intepret
    with pipeline_ready_lock:
        pipeline_ready.append(1)
    if DEVICE_NUMBER > 1:   
        while True:
            # read input
            received_output = b''
            while len(received_output) < batch_size: 
                chunk = prev_socket.recv(com_post_recv_buffer)
                if not chunk:
                    break
                received_output += chunk
            stop_inference_time.append(time.time_ns())
            """
            # transform input from byte to uint
            usable_data = np.frombuffer(received_output, dtype='uint8')
            # reshape input
                # only 1 dimentional array (represent list of classes)
            # interprate
            for i in range(real_limit_set[label_batch]):

                if int(usable_data[i]) == int(label_set[label_batch].flatten()[i]):
                    total_correct += 1
                else:
                    total_wrong += 1
            label_batch += 1
            
            acc = 100*total_correct/(total_correct+total_wrong)
            print("accuracy:\t" + str(acc))
            """


            """
            idx_t = len(stop_inference_time)
            inference_time = (stop_inference_time[idx_t-1]-start_inference_time[idx_t-1])/batch_size
            total_inference_time = 0

            for t in range(idx_t):
                total_inference_time += (stop_inference_time[t]-start_inference_time[t])/batch_size
            average_inference_time = total_inference_time/idx_t
            print("inference_time:\t" + str(inference_time))
            print("average_inference_time:\t" + str(average_inference_time))
            """



                

    elif DEVICE_NUMBER == 1:
        # direct interprate
        while True:
            output_exist = False
            output = None
            if len(fpga_output) > 0:            # output on fpga available
                stop_inference_time.append(time.time_ns())
                with fpga_output_lock:
                    output = fpga_output[0]     # copy output
                    fpga_output.pop(0)
                    output_exist = True
            
            if output_exist:
                pass
                """                                  # interprate
                for i in range(real_limit_set[label_batch]):

                    if int(output.flatten()[i]) == int(label_set[label_batch].flatten()[i]):
                        total_correct += 1
                    else:
                        total_wrong += 1
                label_batch += 1
                acc = 100*total_correct/(total_correct+total_wrong)
                print("accuracy:\t" + str(acc)) 
                output_exist = False
                """

                """
                idx_t = len(stop_inference_time)
                inference_time = (stop_inference_time[idx_t-1]-start_inference_time[idx_t-1])/batch_size
                total_inference_time = 0

                for t in range(idx_t):
                    total_inference_time += (stop_inference_time[t]-start_inference_time[t])/batch_size
                average_inference_time = total_inference_time/idx_t
                print(f"start_inference_time[idx_t-1]:   {start_inference_time[idx_t-1]}")
                print(f"stop_inference_time[idx_t-1]:    {stop_inference_time[idx_t-1]}")
                print(f"inference_time:                  {inference_time}")
                print(f"average_inference_time:          {average_inference_time}")
                """

    else:
        print("the specified number of device is not valid. stopping com_post_recv")