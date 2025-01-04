import numpy as np
def retrive_data_latency(data_shape, freq, batchsize):
    """
    - from fpga to memory
    - not compute yet
    - ref: https://superfastpython.com/benchmark-fastest-way-to-copy-numpy-array/
    """
    latency = 0
    return latency
def decoding_latency(data_shape, freq, batchsize):
    """
    - l = (element * bitwidth * cycle)/f
    - from packed data to normal output shape with data type uint8
    """
    cycle_per_bit = 15
    packedBits = 1 #8 - already consideren when given input is normal shape
    elements = np.prod(data_shape)*batchsize
    latency = (cycle_per_bit*elements*packedBits)/freq
    
    return latency
def transfer_latency(data_shape, freq, batchsize):
    """
    - average 70 MB/s
    - considering each element  data type is unit, so 1 byte for each elements
    """
    average_speed_byte_second = 70*10**6
    elements = np.prod(data_shape)*batchsize
    latency = (1/average_speed_byte_second)*elements
    return latency
def fetch_data_latency(data_shape, freq, batchsize):
    """
    - collect data from the network
    - from byte to uint8
    """
    latency = 0
    return latency
def schaduling_latency(freq):
    """
    - latency due to multi threading (switching)
    """
    latency =0
    return latency
def com_latency(data_shape, cpu_frequency, batchsize, protocol=0):
    """
    - total latency
    """
    latency = 0

    # data from fpga buffer latency
    latency += retrive_data_latency(data_shape, cpu_frequency, batchsize)

    # data decoding latency
    latency += decoding_latency(data_shape, cpu_frequency, batchsize)
    
    # data transfer latency
    latency += transfer_latency(data_shape, cpu_frequency, batchsize)
    
    # data fetch latency
    latency += fetch_data_latency(data_shape, cpu_frequency, batchsize)
    return latency