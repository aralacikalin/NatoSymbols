import os
import argparse
from tqdm import tqdm
from utils import *

def main(labels_in_dir,
         labels_out_dir,
         labels_to_combine,
         shift = False):

    labels_to_nr = read_in_labels('data/labels.txt')

    labels_in_dir = check_dir(labels_in_dir)
    labels_out_dir = check_dir(labels_out_dir)

    if not os.path.exists(labels_out_dir):
        os.makedirs(labels_out_dir)

    transform, shift_loc = read_in_combine(labels_to_combine, labels_to_nr, True)

    if shift:
        transform = fill_missing(transform,labels_to_nr,shift_loc)

    files = os.listdir(labels_in_dir)
    keys = transform.keys()
    for i in tqdm(range(len(files))):
        file = files[i]
        with open(labels_in_dir+'/'+file) as file_in, open(labels_out_dir+'/'+file, 'w') as file_out:
            for line in file_in:
                line_split = line.split(' ')
                if int(line_split[0]) in keys:
                    line_split[0] = str(transform[int(line_split[0])])
                line = " ".join(line_split)
                file_out.write(line)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_in_dir', type=str, default='labels/train', help='Directory where the labels are stored')
    parser.add_argument('--labels_out_dir', type=str, default='labels/train_combined', help='Directory where the combined labels are stored')
    parser.add_argument('--labels_to_combine', type=str, default='combine.txt', help='File where the labels which are to be combined are stored')
    parser.add_argument('--shift', type=bool, default=False, help='When False the new label id will be added. When True the smallest labels nr is taken and the rest of labels are shifted')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))