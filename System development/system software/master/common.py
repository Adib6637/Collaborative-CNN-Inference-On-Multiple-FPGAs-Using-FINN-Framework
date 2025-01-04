from qonnx.core.datatype import DataType
from pynq.pl_server.device import Device
import numpy as np

# FPGA 1

MASTER = 0
DEPENDENT = 1

ROLE = MASTER
DEVICE_NUMBER = 2 ##

dataset_path = "/home/ubuntu/M_Project/Software/multi_fpga_cnn_inference/dataset/SugarWeed/val/"#_sample"

number_of_process = 4  # for syncronization before start the inference to ensure the pipeline is ready 

host_ip = "192.168.178.38"
host_port = 5008
next_host_ip = "192.168.178.40"
next_port = 5008

platform = "zynq-iodma"
batch_size = 16
bitfile = "resizer.bit"
runtime_weight_dir = "runtime_weights/"
devID = 0
device = Device.devices[devID]

''' 
2 module '''
# instantiate FINN accelerator driver and pass batch size and bitfile
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['BINARY']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 32, 32, 3)],
    "oshape_normal" : [(1, 5, 5, 128)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 32, 32, 3, 1)],
    "oshape_folded" : [(1, 5, 5, 1, 128)],
    "ishape_packed" : [(1, 32, 32, 3, 1)],
    "oshape_packed" : [(1, 5, 5, 1, 16)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}


''' 1 module 
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['BINARY']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 32, 32, 3)],
    "oshape_normal" : [(1, 1)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 32, 32, 3, 1)],
    "oshape_folded" : [(1, 1, 1)],
    "ishape_packed" : [(1, 32, 32, 3, 1)],
    "oshape_packed" : [(1, 1, 1)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}'''


accel_ibuf_packed_device_shape = (batch_size, 32, 32, 3, 1)
com_post_recv_buffer = np.prod(io_shape_dict["ishape_normal"][0])*batch_size
com_fetch_input_recv_buffer = batch_size # fow now we focus on 1 bit model