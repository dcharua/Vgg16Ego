import sys, h5py
import cv2, glob
from multiprocessing import Pool
import numpy as np
import os

hf = h5py.File('data4torch.h5', "w")
train_paths = glob.glob('data/static/02/train/*/*.jpg')
def process_image(impath):
    im = cv2.imread(impath)
    im = cv2.resize(im, (150,150))
    im = im.transpose()
    return im

# Class dictionary
label_dict = {'00': 1, '01': 2, '02': 3, '03': 4, '04': 5, '05': 6,
              '06': 7, '07': 8, '08': 9, '09': 10, '10': 11, '11': 12,
              '12': 13, '13': 14, '14': 15, '15': 16, '16': 17, '17': 18,
              '18': 19, '19': 20, '20': 21}

def get_labels(impath):
    label = impath.split('/')[4]
    return label_dict[label]

p = Pool(4) # set this to number of cores you have
data = np.array(p.map(process_image, train_paths)).astype(np.float32)
labels = np.array(p.map(get_labels, train_paths)).astype(np.float32)

hf.create_dataset('data', data=data)
hf.create_dataset('labels', data=labels)
hf.close()
