# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--src-dir', type=str, default='data/ffhq2anime/trainA', help='dataset_dir')
    parser.add_argument('--tgt-dir', type=str, default='data/ffhq2anime/trainA/smooth', help='dataset_dir')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()

def resize_img(src_dir, tgt_dir, img_size) :
    
    file_list = glob(f'{src_dir}/*.*')
    save_dir = f'{tgt_dir}'

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        bgr_img = cv2.resize(bgr_img, (img_size, img_size))

        cv2.imwrite(os.path.join(save_dir, file_name), bgr_img)

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    resize_img(args.src_dir, args.tgt_dir, args.img_size)


if __name__ == '__main__':
    main()
