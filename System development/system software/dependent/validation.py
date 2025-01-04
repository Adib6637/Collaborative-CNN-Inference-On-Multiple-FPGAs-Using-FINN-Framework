import torchvision.datasets as datasets
import torchvision.transforms as transforms
from qonnx.core.datatype import DataType
from tqdm import tqdm
import time
import os
import sys
import torch
import numpy as np

print("\nGetting The driver")
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
}

from util.driver_base import FINNExampleOverlay
print("\nPreparing the accelerator")
driver = FINNExampleOverlay(
    bitfile_name="resizer.bit", #""bitfile,
    platform="zynq-iodma", #platform,
    io_shape_dict=io_shape_dict,
    batch_size=16, #bsize,
    runtime_weight_dir="runtime_weights/",
)


print("\nPreparing the dataset")

transform_d = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

dataset_root = "/home/ubuntu/M_Project/Software/multi_fpga_cnn_inference/dataset/SugarWeed"
path = f"{dataset_root}/val/"
test_dataset = datasets.ImageFolder( path , transform=transform_d)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

print("\nStart validation")
input_set = []
label_set = []
real_limit_set = []
timings = [] 
timings_avg= [] 

databuf = []
labelbuf = []

count = 0
tok = 0
nok = 0

real_limit = 0

b_n = 0

for data, labels in tqdm(testloader):
    data_np = data.numpy()  # shape (16,3,32,32)
    labels_np = labels.numpy()

    if data_np.shape[0] != 16:
        continue

    data_np = data_np.transpose(0,2,3,1) # shape (16,32,32,3)
    ishape = (16, 32, 32, 3, 1)
    data_np = data_np.reshape(ishape)
    data_np = data_np * 255
    data_np = data_np.astype(np.uint8)
    #input_set.append(data_np)
    #label_set.append(labels_np)
    #real_limit_set.append(16) 
    
    ok = 0
    nok = 0
    
    b_n += 1
    ibuf_normal = data_np.reshape(driver.ibuf_packed_device[0].shape)
    exp = labels
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)   

    ret = np.bincount(obuf_normal.flatten() == exp.flatten())
    nok += ret[0]
    ok += ret[1]
    tok += ret[1]
    print("batch %d  : total OK %d NOK %d" % (b_n, ok, nok))

acc = 100.0 * tok / (b_n*16)
print("Final accuracy: %f" % acc)
############################################################








"""

    ibuf_normal = data_ready.reshape(driver.ibuf_packed_device[0].shape)
    driver.copy_input_data_to_device(ibuf_normal)
    start = time.time()
    driver.execute_on_buffers()
    end = time.time()


    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)
    ret = [0, 0]


    for i  in range(real_limit):#0,(len(obuf_normal.flatten())-1)):
        if (obuf_normal.flatten()[i] != exp.flatten()[i]):
            ret[0] += 1
        else:
            ret[1] += 1
    nok += ret[0]
    ok += ret[1]

    print("batch %d / %d : total OK %d NOK %d" % (count  + 1, len(testloader), ok, nok))
    count+=1
    curr_time = (end - start)/16
    timings.append(curr_time)

accuracy = 100.0 * ok / (nok+ok)
avg_inf_t = np.average(timings)
print("Final accuracy: %f" % accuracy)
print("Mean Inference Time:"+str(avg_inf_t))
print("\nValidation finished!")
exit()
"""




"""
#remove excess
R = all_data_np.shape[0]%bsize
if R != 0:
    all_data_np_t = all_data_np_t[:-R, :, :, :]
    all_labels_np_t = all_labels_np[:-R,]
all_data_np_t = all_data_np_t * gain_val
all_data_np_t = all_data_np_t.astype(dtype)
total = all_data_np.shape[0]
n_batches = int(total / bsize)
test_images = all_data_np_t.reshape(n_batches, bsize, -1)
test_labels = all_labels_np_t.reshape(n_batches, bsize)

return test_images, test_labels, total, n_batches


        for data, target in testloader:
            output = model(data)



            loss = loss_fn(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            correct += (output.argmax(1) == target).type(torch.float).sum().item()
            dur.append(endtime-startTime)

        correct /= size
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_dur.append(np.average(dur))
###############################################################################################################
###############################################################################################################




print("\nStart validation")
timings=np.zeros((total,1))
count = 0
ok = 0
nok = 0
for batch_idx  in tqdm(range(n_batches)):  
    ibuf_normal = test_images[batch_idx].reshape(driver.ibuf_packed_device[0].shape)
    exp = test_labels[batch_idx]
    driver.copy_input_data_to_device(ibuf_normal)
    # measure time
    start = time.time()
    driver.execute_on_buffers()
    end = time.time()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    
    print("copy_output_data_from_device")
    driver.copy_output_data_from_device(obuf_normal)
    ret = [0, 0]

    print("in range(0,len(obuf_normal.flatten())")
    for i  in range(0,(len(obuf_normal.flatten())-1)):
        print("if (obuf_normal.flatten()[i] != exp.flatten()[i]):")
        if (obuf_normal.flatten()[i] != exp.flatten()[i]):
            ret[0] += 1
        else:
            ret[1] += 1
    nok += ret[0]
    ok += ret[1]
    print_intermidiate_progress = False
    print("print_intermidiate_progress")
    if(print_intermidiate_progress):
        print("batch %d / %d : total OK %d NOK %d" % (batch_idx  + 1, n_batches, ok, nok))
    curr_time = end - start
    timings[batch_idx] = curr_time 
accuracy = 100.0 * ok / (total)
avg_inf_t = np.average(timings)#np.sum(timings) / n_batches
print("Final accuracy: %f" % accuracy)
print("Mean Inference Time:"+str(avg_inf_t))
print("\nValidation finished!")
exit()"""