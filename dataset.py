import cv2
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

SIZE_TRAIN = 92
SIZE_TEST = 92
IM_SIZE = 1024
DATASET = 'Fluo-N2DH-GOWT1'


X_TRAIN_PATH = lambda s: f'../data/{DATASET}/{s:02}/t'
Y_TRAIN_PATH = lambda s: f'../data/{DATASET}/{s:02}_ST/SEG/man_seg'
X_TEST_PATH = lambda s: f'../data/{DATASET}_chal/{DATASET}/{s:02}/t'


def train_generator(setn,b):
    x = []
    y = []
    for i in range(SIZE_TRAIN):
        x.append(np.asarray(cv2.imread(X_TRAIN_PATH(setn)+f"{i:03}"+".tif", -1), dtype=np.uint16).astype(
            dtype=np.float64).reshape([1, IM_SIZE, IM_SIZE]))
        y.append((np.asarray(cv2.imread(
            Y_TRAIN_PATH(setn)+f"{i:03}"+".tif", -1), dtype=np.uint16) != 0).astype(int).reshape([1, IM_SIZE, IM_SIZE]).astype(np.float64))
        if len(x) == b:
            yield (torch.from_numpy(np.array(x).reshape([b, 1, IM_SIZE, IM_SIZE])).float(), torch.from_numpy(np.array(y).reshape([b, 1, IM_SIZE, IM_SIZE])).float())
            x = []
            y = []

def test_generator(setn,b):
    x = []
    for i in range(SIZE_TEST):
        x.append(np.asarray(cv2.imread(X_TEST_PATH(setn)+f"{i:03}"+".tif", -1), dtype=np.uint16).astype(
            dtype=np.float64).reshape([IM_SIZE, IM_SIZE]))
        if len(x) == b:
            yield (torch.from_numpy(np.array(x).reshape([b, 1, IM_SIZE, IM_SIZE])).float())
            x=[]

# a = input("Enter:")

# image = cv2.imread(Y_PATH+a+'.tif', -1)
# image = np.asarray(image, dtype=np.uint16)
# print([list(i) for i in list(image)])
# print((image.min(), image.max(), image.mean()))
# plt.imsave('tiff.png', image.astype(np.uint16))
