# check pytorch installation:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.8")   # please manually install torch 1.8 if Colab changes its default version
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries like numpy, json, cv2 etc.
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2

# Inference with a keypoint detection model
cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

import glob
path ="/home/josyula/Downloads/train_video/*.jpg"
files = glob.glob(path)
files = sorted(files)
i=0
for filename in files:
    im = cv2.imread(filename)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()
    res = img[:, :, ::-1]
    print(cv2.imwrite("/home/josyula/Downloads/train_video/pred"+str(i)+"pred.jpg", res))
    i+=1