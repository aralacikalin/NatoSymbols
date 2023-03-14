from tqdm import tqdm
from utils import inverse, check_dir
from cv2 import imread, imwrite
from os import path, makedirs, listdir

import argparse

def main(images_in_dir,
         images_out_dir,
         threshold=110):
    
    # Check if dir string ends with "/"
    images_in_dir = check_dir(images_in_dir)
    images_out_dir = check_dir(images_out_dir)

    # If output dir does not exists then create one
    if not path.exists(images_out_dir):
        makedirs(images_out_dir)

    # Get the images_files in input dir
    images_files = listdir(images_in_dir)
    
    for i in tqdm(range(len(images_files))):
        img_file = images_files[i]
        img = imread(f'{images_in_dir}/{img_file}')
        img_inverse = inverse(img, threshold)
        imwrite(f'{images_out_dir}/{img_file}',img_inverse)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_in_dir', type=str, default='images/train', help='Directory where the images are stored')
    parser.add_argument('--images_out_dir', type=str, default='images/train_inverse', help='Directory where the inversed images will be stored')
    parser.add_argument('--threshold', type=int, default=110, help='Threshold value for inverse')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))