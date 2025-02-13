"""
@Author: Xiaohan Chen
@Email: cxh_bb@outlook.com
"""

import numpy as np
from scipy.io import loadmat
from PreparData.preprocess import transformation

# datanames in every working conditions
dataname_dict= {0:[97, 105, 118, 130, 169, 185, 197, 209, 222, 234],  # 1797rpm
                1:[98, 119, 186, 223, 106, 170, 210, 131, 198, 235],  # 1772rpm
                2:[99, 120, 187, 224, 107, 171, 211, 132, 199, 236],  # 1750rpm
                3:[100,121, 188, 225, 108, 172, 212, 133, 200, 237]}  # 1730rpm

axis = "_DE_time"
data_length = 2048


def CWRU(datadir, load, labels, window, normalization, backbone, fft):
    """
    loading the hole dataset
    """
    path = datadir + "/CWRU/" + "Drive_end_" + str(load) + "/"
    dataset = {label: [] for label in labels}
    for label in labels:
        fault_type = dataname_dict[load][label]
        if fault_type < 100:
            realaxis = "X0" + str(fault_type) + axis
        else:
            realaxis = "X" + str(fault_type) + axis
        mat_data = loadmat(path+str(fault_type)+".mat")[realaxis]
        start, end = 0, data_length

        # set the endpoint of data sequence
        endpoint = mat_data.shape[0]

        # split the data and transformation
        while end < endpoint:
            sub_data = mat_data[start : end].reshape(-1,)

            sub_data = transformation(sub_data, fft, normalization, backbone)

            dataset[label].append(sub_data)
            start += window
            end += window
        
        dataset[label] = np.array(dataset[label], dtype="float32")

    return dataset

def CWRUloader(args, load, label_set, number="all"):
    """
    args: arguments
    number: the numbers of training samples, "all" or specific numbers (string type)
    """
    dataset = CWRU(args.datadir, load, label_set, args.window, args.normalization, args.backbone, args.fft)

    DATA, LABEL = [], []

    if number == "all":
        counter = []
        for key in dataset.keys():
            counter.append(dataset[key].shape[0])
        datan = min(counter) # choosing the min value as the sample size per class
        for key in dataset.keys():
            LABEL.append(np.tile(key, datan))
            DATA.append(dataset[key][:datan])
    else:
        datan = int(number)
        for key in dataset.keys():
            LABEL.append(np.tile(key, datan))
            DATA.append(dataset[key][:datan])
    
    DATA, LABEL = np.array(DATA, dtype="float32"), np.array(LABEL, dtype="int32")

    return DATA, LABEL