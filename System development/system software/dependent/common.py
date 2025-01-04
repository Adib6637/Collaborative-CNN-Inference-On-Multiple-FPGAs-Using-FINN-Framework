from qonnx.core.datatype import DataType
from pynq.pl_server.device import Device
import numpy as np

# FPGA 1

MASTER = 0
DEPENDENT = 1

ROLE = DEPENDENT
DEVICE_NUMBER = 2 ##

dataset_path = "/home/ubuntu/M_Project/Software/multi_fpga_cnn_inference/dataset/SugarWeed/val/"#_sample"

number_of_process = 4  # for syncronization before start the inference to ensure the pipeline is ready 

host_ip = "192.168.178.40"
host_port = 5008
next_host_ip = "192.168.178.38"
next_port = 5008

platform = "zynq-iodma"
batch_size = 16
bitfile = "resizer.bit"
runtime_weight_dir = "runtime_weights/"
devID = 0
device = Device.devices[devID]


# instantiate FINN accelerator driver and pass batch size and bitfile
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['BINARY']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 5, 5, 128)],
    "oshape_normal" : [(1, 1)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 5, 5, 4, 32)],
    "oshape_folded" : [(1, 1, 1)],
    "ishape_packed" : [(1, 5, 5, 4, 32)],
    "oshape_packed" : [(1, 1, 1)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}

#accel_ibuf_packed_device_shape = (batch_size, 5, 5, 4, 32)
accel_ibuf_packed_device_shape = (batch_size, 5, 5, 4, 32)
com_post_recv_buffer = np.prod(io_shape_dict["ishape_normal"][0])*batch_size
com_fetch_input_recv_buffer = np.prod(io_shape_dict["ishape_normal"][0])*batch_size