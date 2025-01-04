import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

def dataset_transforms(dataset_path, accel_ibuf_packed_device_shape):
    # load dataset
    transform_d = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.ImageFolder(dataset_path, transform=transform_d)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    # change the dataset format and shape
    input_set = []
    label_set = []
    real_limit_set = []

    data_buffer = []
    label_buffer = []
    for data, labels in tqdm(test_loader):
        # Convert batch_data and batch_labels to NumPy arrays
        data_np = data.numpy()
        labels_np = labels.numpy()
        data_buffer.append(data_np)
        label_buffer.append(labels_np)
        data_concatenate = np.concatenate(data_buffer, axis=0)
        labels_concatenate = np.concatenate(label_buffer, axis=0)
        data_buffer = []
        label_buffer = []
        data_concatenate_t = data_concatenate.transpose(0, 2, 3, 1)
        data_concatenate_t = data_concatenate_t * 255
        data_concatenate_t = data_concatenate_t.astype(np.uint8)
        
        real_limit = data_concatenate_t.shape[0]
        real_limit_set.append(real_limit)                                           # store the length of each batch

        data_ready = data_concatenate_t.reshape(data_concatenate_t.shape[0], -1)
        exp = labels_concatenate
        # change the last batch size to fit with pack size
        if data_ready.shape[0] != accel_ibuf_packed_device_shape:
            # continue
            new_shape = (16, 3072)
            new_array = np.zeros(new_shape, dtype=np.uint8)
            new_array[:data_ready.shape[0], :] = data_ready
            data_ready = new_array
            #print(data_ready.shape)

        input_buf_normal = data_ready.reshape(accel_ibuf_packed_device_shape)
        input_set.append(input_buf_normal)                                          # store the transformed inputs
        label_set.append(exp)

    
    
    return input_set, label_set, real_limit_set # store the transformed labels

    # TODO: store all the reshape input into a file
    #compiled_set = np.array([input_set, label_set, real_limit_set])
    #np.save(output_path + "/compiled_set.npy", compiled_set)
