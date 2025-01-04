from common import *
import numpy as np
import socket

def receive_input(fetch_input_receive,fetch_input_receive_lock,fpga_input,fpga_input_lock, pipeline_ready, pipeline_ready_lock):
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prev_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    print("preparing the host socket")
    # initialize localhost port
    host_socket.bind((host_ip, host_port))

    print("listening ...")
    host_socket.listen(1)
    prev_socket, prev_address = host_socket.accept() 
    print("connected to the pervious host")
    with pipeline_ready_lock:
        pipeline_ready.append(1)
    while True:
        input_data = b''
        while len(input_data) < com_fetch_input_recv_buffer:
            chunk = prev_socket.recv(com_fetch_input_recv_buffer)
            if not chunk:
                break
            input_data += chunk
        uint8_data = np.frombuffer(input_data, dtype=np.uint8)

        ######################################################   !warning: this is hard coded  ###################################
        target_shape = (16,1,5,5,128)#io_shape_dict["ishape_normal"][0][0]
        valid_input_raw = uint8_data.reshape(target_shape)
        valid_input_transpose = np.transpose(valid_input_raw, (0, 3, 1, 2, 4))
        ##########################################################################################################################

        valid_input = valid_input_transpose.reshape(accel_ibuf_packed_device_shape)


        while len(fpga_input) > 0:                              # wait for input buffer to empty
            pass
                                                                # if input buffer in fpga is empty
        with fpga_input_lock:
            fpga_input.append(valid_input)                      # put the input into buffer



                
        








