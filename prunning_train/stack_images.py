import cv2
import numpy as np
from tifffile import imwrite
from tifffile import imread

import os

path_to_img = "/home/josyula/Documents/DataAndModels/pruning_training/train/"
path_to_write = "/home/josyula/Documents/DataAndModels/pruning_training/stacked/"
temp=np.zeros((480,640,4))
for i in range(618):
    actual = path_to_img+str(i)+".json"
    os.system("cp "+actual+" "+path_to_write)
    # mask = str(i)+"_mask.png"
    # flow = str(i)+"_flow.png"
    # # im1 = cv2.imread(path_to_img+actual)
    # im2 = cv2.imread(path_to_img+mask)
    # gray_image = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # im3 = cv2.imread(path_to_img+flow)
    # temp[:,:,:3 ] = im3
    # temp[:,:,3] = gray_image
    # print(imwrite(path_to_write+str(i)+".tiff", temp))
# x = cv2.imread(path_to_write+"0.png")
# y = cv2.imread(path_to_img+"0_flow.png")
# z = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
# print(x.shape)