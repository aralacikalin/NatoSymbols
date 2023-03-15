import numpy as np

def read_in_labels(file):
    labels_to_nr = {}
    i = 0
    with open(file) as f:
        for line in f:
            labels_to_nr[line.strip('\n')] = i
            i += 1
    return labels_to_nr

def read_in_combine(labels_to_combine, labels_to_nr, shift):
    transform = {}
    labels_numerical_values = []
    labels_in_group=[]
    labels_in_group.append([])

    with open(labels_to_combine) as file:
        if shift:
            i = 0
        else:
            i = len(labels_to_nr.keys())
        while True:
            line = file.readline()
            if not line:
                break
            if line != "\n":
                label = line.strip("\n")
                labels_in_group[-1].append(label)
                if shift:
                    transform[labels_to_nr[label]] = i
                else:
                    transform[labels_to_nr[label]] = i
                labels_numerical_values.append(labels_to_nr[label])
            else:
                i += 1
                labels_in_group.append([])
    
    shift_loc = None
    if shift:
        shift_loc = []
        pointer_left = 0
        pointer_right = 0
        for group in labels_in_group:
            pointer_right += len(group)
            value = np.min(labels_numerical_values[pointer_left:pointer_right])
            shift_loc.append(value)
            for label in group:
                transform[labels_to_nr[label]] = value
            pointer_left = pointer_right
                
    return transform, shift_loc

def fill_missing(transform, labels_to_nr, no_shift_loc):
    transform_shifted = {}
    keys = transform.keys()
    shift = 0
    no_shift_loc.sort()
    for i in range(len(labels_to_nr)):
        if i not in keys:
            while(i+shift in no_shift_loc):
                shift += 1
            transform_shifted[i] = i + shift
        else:
            transform_shifted[i] = transform[i]
            if i != no_shift_loc[0]:
                shift -= 1
    return transform_shifted

def check_dir(dir):
    if dir[-1] == '/':
        dir = dir[:-1]
    return dir

def inverse(img, binary_threshold=110):
    loc = img <= binary_threshold
    img[img > binary_threshold] = 0
    img[loc] = 255
    return img